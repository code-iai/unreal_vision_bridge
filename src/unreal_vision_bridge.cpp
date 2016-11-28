/**
 * Copyright 2014 University of Bremen, Institute for Artificial Intelligence
 * Author: Thiemo Wiedemeyer <wiedemeyer@cs.uni-bremen.de>
 *
 * This file is part of pico_flexx_driver.
 *
 * pico_flexx_driver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * pico_flexx_driver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with pico_flexx_driver.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <ros/ros.h>
#include <ros/console.h>
#include <nodelet/nodelet.h>

#include <tf/transform_broadcaster.h>

#include <std_msgs/Header.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

#include <unreal_vision_bridge/ObjectColorMap.h>

#if defined(__linux__)
#include <sys/prctl.h>
#elif defined(__APPLE__)
#include <pthread.h>
#endif

static inline void setThreadName(const std::string &name)
{
#if defined(__linux__)
  prctl(PR_SET_NAME, name.c_str());
#elif defined(__APPLE__)
  pthread_setname_np(name.c_str());
#endif
}

#define UV_DEFAULT_NS   "unreal_vision"
#define UV_TF_OPT_FRAME "_optical_frame"
#define UV_TOPIC_INFO   "/camera_info"
#define UV_TOPIC_COLOR  "/image_color"
#define UV_TOPIC_DEPTH  "/image_depth"
#define UV_TOPIC_OBJECT "/image_object"
#define UV_TOPIC_MAP    "/object_color_map"

#define NO_COLOR   "\033[0m"
#define FG_BLACK   "\033[30m"
#define FG_RED     "\033[31m"
#define FG_GREEN   "\033[32m"
#define FG_YELLOW  "\033[33m"
#define FG_BLUE    "\033[34m"
#define FG_MAGENTA "\033[35m"
#define FG_CYAN    "\033[36m"

#define OUT_FUNCTION(NAME) ([](const std::string &name)\
{ \
  size_t end = name.rfind('(');\
  if(end == std::string::npos) end = name.size();\
  size_t begin = 1 + name.rfind(' ', end);\
  return name.substr(begin, end - begin);\
}(NAME))
#define OUT_AUX(FUNC_COLOR, MSG_COLOR, STREAM, MSG) STREAM << FUNC_COLOR "[" << OUT_FUNCTION(__PRETTY_FUNCTION__) << "] " MSG_COLOR << MSG << NO_COLOR << std::endl

#define OUT_DEBUG(msg) OUT_AUX(FG_BLUE, NO_COLOR, std::cout, msg)
#define OUT_INFO(msg) OUT_AUX(FG_GREEN, NO_COLOR, std::cout, msg)
#define OUT_WARN(msg) OUT_AUX(FG_YELLOW, FG_YELLOW, std::cout, msg)
#define OUT_ERROR(msg) OUT_AUX(FG_RED, FG_RED, std::cerr, msg)

class UnrealVisionBridge
{
private:
  struct Vector
  {
    float x;
    float y;
    float z;
  };

  struct Quaternion
  {
    float x;
    float y;
    float z;
    float w;
  };

  struct PacketHeader
  {
    uint32_t size;
    uint32_t sizeHeader;
    uint32_t mapEntries;
    uint32_t width;
    uint32_t height;
    uint64_t timestampCapture;
    uint64_t timestampSent;
    float fieldOfViewX;
    float fieldOfViewY;
    Vector translation;
    Quaternion rotation;
  };

  struct MapEntry
  {
    uint32_t size;
    uint8_t r;
    uint8_t g;
    uint8_t b;
    char firstChar;
  };

  struct Packet
  {
    PacketHeader header;
    uint8_t *pColor, *pDepth, *pObject, *pMap;
    size_t sizeColor, sizeDepth, sizeObject;
  } packet;

  enum Topics
  {
    CAMERA_INFO = 0,
    COLOR,
    DEPTH,
    OBJECT,
    MAP,
    COUNT
  };

  const size_t sizeRGB;
  const size_t sizeFloat;

  ros::NodeHandle nh, priv_nh;
  tf::TransformBroadcaster broadcaster;
  std::vector<ros::Publisher> publisher;
  std::vector<bool> status;
  std::thread receiver, transmitter;
  std::mutex lockBuffer;
  std::condition_variable cvNewData;
  bool running, newData;

  std::vector<uint8_t> bufferComplete, bufferActive;

  std::string baseName, baseNameTF, address;
  uint16_t port;
  int connection;

public:
  UnrealVisionBridge(const ros::NodeHandle &nh = ros::NodeHandle(), const ros::NodeHandle &priv_nh = ros::NodeHandle("~"))
    : sizeRGB(3 * sizeof(uint8_t)), sizeFloat(sizeof(float)), nh(nh), priv_nh(priv_nh), running(false), newData(false)
  {
    int portNumber, queueSize;
    priv_nh.param("base_name", baseName, std::string(UV_DEFAULT_NS));
    priv_nh.param("base_name_tf", baseNameTF, baseName);
    priv_nh.param("address", address, std::string("192.168.100.160"));
    priv_nh.param("port", portNumber, 10000);
    priv_nh.param("queue_size", queueSize, 5);

    port = (uint16_t)std::min(std::max(portNumber, 0), 0xFFFF);

    OUT_INFO("parameter:" << std::endl
             << "   base_name: " FG_CYAN << baseName << NO_COLOR << std::endl
             << "base_name_tf: " FG_CYAN << baseNameTF << NO_COLOR << std::endl
             << "     address: " FG_CYAN << address << NO_COLOR << std::endl
             << "        port: " FG_CYAN << port << NO_COLOR << std::endl
             << "  queue_size: " FG_CYAN << queueSize << NO_COLOR);

    setTopics(baseName, queueSize);

    const size_t bufferSize = 1024 * 1024 * 10;
    bufferComplete.resize(bufferSize);
    bufferActive.resize(bufferSize);
  }

  ~UnrealVisionBridge()
  {
    stop();
  }

  bool start()
  {
    struct sockaddr_in serverAddress;

    OUT_INFO("creating socket.");
    connection = socket(AF_INET, SOCK_STREAM, 0);
    if(connection < 0)
    {
      OUT_ERROR("could not open socket");
      return false;
    }

    bzero((char *) &serverAddress, sizeof(serverAddress));
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = inet_addr(address.c_str());
    serverAddress.sin_port = htons(port);

    OUT_INFO("connecting to server.");
    while(ros::ok())
    {
      if(connect(connection, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) >= 0)
      {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    OUT_INFO("starting receiver and transmitter threads.");
    running = true;
    transmitter = std::thread(&UnrealVisionBridge::transmit, this);
    receiver = std::thread(&UnrealVisionBridge::receive, this);
    return true;
  }

  void stop()
  {
    running = false;
    if(transmitter.joinable())
    {
      transmitter.join();
    }
    if(receiver.joinable())
    {
      receiver.join();
    }
  }

private:
  void setTopics(const std::string &baseName, const int32_t queueSize)
  {
    publisher.resize(COUNT);
    status.resize(COUNT, false);
    ros::SubscriberStatusCallback cb = boost::bind(&UnrealVisionBridge::callbackTopicStatus, this);
    publisher[CAMERA_INFO] = nh.advertise<sensor_msgs::CameraInfo>(baseName + UV_TOPIC_INFO, queueSize, cb, cb);
    publisher[COLOR] = nh.advertise<sensor_msgs::Image>(baseName + UV_TOPIC_COLOR, queueSize, cb, cb);
    publisher[DEPTH] = nh.advertise<sensor_msgs::Image>(baseName + UV_TOPIC_DEPTH, queueSize, cb, cb);
    publisher[OBJECT] = nh.advertise<sensor_msgs::Image>(baseName + UV_TOPIC_OBJECT, queueSize, cb, cb);
    publisher[MAP] = nh.advertise<unreal_vision_bridge::ObjectColorMap>(baseName + UV_TOPIC_MAP, queueSize, cb, cb);
  }

  void callbackTopicStatus()
  {
    lockBuffer.lock();
    bool clientsConnected = false;
    for(size_t i = 0; i < COUNT; ++i)
    {
      status[i] = publisher[i].getNumSubscribers() > 0;
      clientsConnected = clientsConnected || status[i];
    }
    lockBuffer.unlock();
  }

  void receive()
  {
    setThreadName("receiver");
    const size_t minSize = std::min((size_t)1024, bufferActive.size());
    uint8_t *pPackage = &bufferActive[0];
    PacketHeader header;
    size_t written = 0, left = minSize;

    OUT_INFO("receiver started.");
    while(ros::ok() && running)
    {
      ssize_t bytesRead = read(connection, pPackage + written, left);
      if(bytesRead <= 0)
      {
        OUT_ERROR("could not read from socket.");
        break;
      }

      left -= bytesRead;
      written += bytesRead;

      if(header.size == 0 && written > sizeof(PacketHeader))
      {
        header = *reinterpret_cast<PacketHeader *>(pPackage);

        if(bufferActive.size() < header.size)
        {
          // make it 1 mb bigger that the actual package size, so that the buffer does not need to be resized often
          bufferActive.resize(header.size + 1024 * 1024);
          pPackage = &bufferActive[0];
          OUT_INFO("resized buffer to: " << bufferActive.size());
        }
        left = header.size - written;
      }

      if(header.size != 0 && left == 0)
      {
        uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        OUT_INFO("package complete. delay: " << (now - header.timestampSent) / 1000000.0 << " ms.");
        /*OUT_INFO("header size: " << header.size);
        OUT_INFO("header sizeHeader: " << header.sizeHeader);
        OUT_INFO("header mapEntries: " << header.mapEntries);
        OUT_INFO("header width: " << header.width);
        OUT_INFO("header height: " << header.height);
        OUT_INFO("header timestampCapture: " << header.timestampCapture);
        OUT_INFO("header timestampSent: " << header.timestampSent);*/

        if(header.sizeHeader != sizeof(PacketHeader))
        {
          OUT_ERROR("package header size does not match expectations: " << sizeof(PacketHeader) << " received: " << header.sizeHeader);
        }

        lockBuffer.lock();
        bufferActive.swap(bufferComplete);

        packet.header = header;
        packet.sizeColor = header.width * header.height * sizeRGB;
        packet.sizeDepth = header.width * header.height * sizeFloat;
        packet.sizeObject = header.width * header.height * sizeRGB;
        packet.pColor = &bufferComplete[sizeof(PacketHeader)];
        packet.pDepth = packet.pColor + packet.sizeColor;
        packet.pObject = packet.pDepth + packet.sizeDepth;
        packet.pMap = packet.pObject + packet.sizeColor;
        newData = true;

        pPackage = &bufferActive[0];
        header.size = 0;
        written = 0;
        left = minSize;
        lockBuffer.unlock();
        cvNewData.notify_one();
      }
    }
    close(connection);
    running = false;
    ros::shutdown();
    OUT_INFO("receiver stopped.");
  }

  void transmit()
  {
    setThreadName("transmitter");
    sensor_msgs::CameraInfoPtr msgCameraInfo(new sensor_msgs::CameraInfo);
    sensor_msgs::ImagePtr msgColor(new sensor_msgs::Image), msgDepth(new sensor_msgs::Image), msgObject(new sensor_msgs::Image);
    unreal_vision_bridge::ObjectColorMapPtr msgMap(new unreal_vision_bridge::ObjectColorMap);
    std::unique_lock<std::mutex> lock(lockBuffer);

    OUT_INFO("transmitter started.");
    while(ros::ok() && running)
    {
      if(!cvNewData.wait_for(lock, std::chrono::milliseconds(300), [this] { return this->newData; }))
      {
        continue;
      }

      extractData(msgCameraInfo, msgColor, msgDepth, msgObject, msgMap);
      publish(msgCameraInfo, msgColor, msgDepth, msgObject, msgMap);
      newData = false;

      OUT_INFO("images sent.");
    }
    running = false;
    ros::shutdown();
    OUT_INFO("transmitter stopped.");
  }

  void extractData(sensor_msgs::CameraInfoPtr &msgCameraInfo, sensor_msgs::ImagePtr &msgColor, sensor_msgs::ImagePtr &msgDepth, sensor_msgs::ImagePtr &msgObject, unreal_vision_bridge::ObjectColorMapPtr &msgMap) const
  {
    std_msgs::Header header;
    header.frame_id = baseNameTF + UV_TF_OPT_FRAME;
    header.seq = 0;

    uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    header.stamp.fromNSec((ros::Time::now() - ros::Time().fromNSec(now - packet.header.timestampCapture)).toNSec());

    //OUT_INFO("Translation: (" << packet.header.translation.x << " " << packet.header.translation.y << " " << packet.header.translation.z << ")");
    //OUT_INFO("Rotation: (" << packet.header.rotation.x << " " << packet.header.rotation.y << " " << packet.header.rotation.z << " " << packet.header.rotation.w << ")");

    if(status[CAMERA_INFO])
    {
      setCameraInfo(msgCameraInfo);
    }

    if(status[COLOR] || status[DEPTH] || status[OBJECT])
    {
      msgObject->header = msgDepth->header = msgColor->header = header;
      msgObject->height = msgDepth->height = msgColor->height = packet.header.height;
      msgObject->width = msgDepth->width = msgColor->width = packet.header.width;
      msgObject->is_bigendian = msgDepth->is_bigendian = msgColor->is_bigendian = false;
    }

    if(status[COLOR])
    {
      msgColor->encoding = sensor_msgs::image_encodings::BGR8;
      msgColor->step = (uint32_t)(sizeRGB * packet.header.width);
      msgColor->data.resize(packet.sizeColor);
      memcpy(&msgColor->data[0], packet.pColor, packet.sizeColor);
    }
    if(status[DEPTH])
    {
      msgDepth->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
      msgDepth->step = (uint32_t)(sizeFloat * packet.header.width);
      msgDepth->data.resize(packet.sizeDepth);
      memcpy(&msgDepth->data[0], packet.pDepth, packet.sizeDepth);
    }
    if(status[OBJECT])
    {
      msgObject->encoding = sensor_msgs::image_encodings::BGR8;
      msgObject->step = (uint32_t)(sizeRGB * packet.header.width);
      msgObject->data.resize(packet.sizeObject);
      memcpy(&msgObject->data[0], packet.pObject, packet.sizeObject);
    }
    if(status[MAP])
    {
      const size_t SizeEntryHeader = sizeof(uint32_t) + 3 * sizeof(uint8_t);
      uint8_t *it = packet.pMap;
      for(uint32_t i = 0; i < packet.header.mapEntries; ++i)
      {
        const MapEntry *entry = reinterpret_cast<MapEntry *>(it);
        std_msgs::ColorRGBA color;
        color.r = entry->r;
        color.g = entry->g;
        color.b = entry->b;
        color.a = 0;

        std_msgs::String name;
        name.data = std::string(&entry->firstChar, entry->size - SizeEntryHeader);

        //OUT_INFO("map entry: " << name.data << " : " << color.r << " " << color.g << " " << color.b);

        msgMap->colors.push_back(color);
        msgMap->names.push_back(name);

        it += entry->size;
      }
    }
  }
  void setCameraInfo(sensor_msgs::CameraInfoPtr msgCameraInfo) const
  {
    const double halfFieldOfViewX = packet.header.fieldOfViewX * M_PI / 360.0;
    const double halfFieldOfViewY = packet.header.fieldOfViewY * M_PI / 360.0;
    double axisMultiplierX = 1.0;
    double axisMultiplierY = 1.0;

    if(packet.header.width > packet.header.height)
    {
      // if the viewport is wider than it is tall
      axisMultiplierY = packet.header.width / (double)packet.header.height;
    }
    else
    {
      // if the viewport is taller than it is wide
      axisMultiplierX = packet.header.height / (double)packet.header.width;
    }

    msgCameraInfo->height = packet.header.width;
    msgCameraInfo->width = packet.header.height;

    msgCameraInfo->K.assign(0.0);
    msgCameraInfo->K[0] = axisMultiplierX / std::tan(halfFieldOfViewX);
    msgCameraInfo->K[2] = packet.header.width / 2.0;
    msgCameraInfo->K[4] = axisMultiplierY / std::tan(halfFieldOfViewY);
    msgCameraInfo->K[5] = packet.header.height / 2.0;
    msgCameraInfo->K[8] = 1;

    msgCameraInfo->R.assign(0.0);
    msgCameraInfo->R[0] = 1;
    msgCameraInfo->R[4] = 1;
    msgCameraInfo->R[8] = 1;

    msgCameraInfo->P.assign(0.0);
    msgCameraInfo->P[0] = msgCameraInfo->K[0];
    msgCameraInfo->P[2] = msgCameraInfo->K[2];
    msgCameraInfo->P[5] = msgCameraInfo->K[4];
    msgCameraInfo->P[6] = msgCameraInfo->K[5];
    msgCameraInfo->P[10] = 1;

    msgCameraInfo->distortion_model = "plumb_bob";
    msgCameraInfo->D.resize(5, 0.0);
  }

  void publish(sensor_msgs::CameraInfoPtr &msgCameraInfo, sensor_msgs::ImagePtr &msgColor, sensor_msgs::ImagePtr &msgDepth, sensor_msgs::ImagePtr &msgObject, unreal_vision_bridge::ObjectColorMapPtr &msgMap)
  {
    tf::Vector3 translation(packet.header.translation.x, packet.header.translation.y, packet.header.translation.z);
    tf::Quaternion rotation(packet.header.rotation.x, packet.header.rotation.y, packet.header.rotation.z, packet.header.rotation.w);
    broadcaster.sendTransform(tf::StampedTransform(tf::Transform(rotation, translation), msgCameraInfo->header.stamp, "map", baseNameTF + UV_TF_OPT_FRAME));

    if(status[CAMERA_INFO])
    {
      publisher[CAMERA_INFO].publish(msgCameraInfo);
      msgCameraInfo = sensor_msgs::CameraInfoPtr(new sensor_msgs::CameraInfo);
    }
    if(status[COLOR])
    {
      publisher[COLOR].publish(msgColor);
      msgColor = sensor_msgs::ImagePtr(new sensor_msgs::Image);
    }
    if(status[DEPTH])
    {
      publisher[DEPTH].publish(msgDepth);
      msgDepth = sensor_msgs::ImagePtr(new sensor_msgs::Image);
    }
    if(status[OBJECT])
    {
      publisher[OBJECT].publish(msgObject);
      msgObject = sensor_msgs::ImagePtr(new sensor_msgs::Image);
    }
    if(status[MAP])
    {
      publisher[MAP].publish(msgMap);
      msgMap = unreal_vision_bridge::ObjectColorMapPtr(new unreal_vision_bridge::ObjectColorMap);
    }
  }
};

class UnrealVisionNodelet : public nodelet::Nodelet
{
private:
  UnrealVisionBridge *unrealVisionBridge;

public:
  UnrealVisionNodelet() : Nodelet(), unrealVisionBridge(NULL)
  {
  }

  ~UnrealVisionNodelet()
  {
    if(unrealVisionBridge)
    {
      unrealVisionBridge->stop();
      delete unrealVisionBridge;
    }
  }

  virtual void onInit()
  {
    unrealVisionBridge = new UnrealVisionBridge(getNodeHandle(), getPrivateNodeHandle());
    unrealVisionBridge->start();
  }
};

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(UnrealVisionNodelet, nodelet::Nodelet)

int main(int argc, char **argv)
{
  setThreadName("main thread");
  ros::init(argc, argv, UV_DEFAULT_NS);

  UnrealVisionBridge unrealVisionBridge;
  if(!unrealVisionBridge.start())
  {
    return -1;
  }
  ros::spin();
  unrealVisionBridge.stop();
  return 0;
}
