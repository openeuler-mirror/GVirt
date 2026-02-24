/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_SOCK_H_
#define _XLITE_SOCK_H_

#include <cstdint>
#include <vector>

class XSock
{
public:
    XSock(uint32_t rankId, uint32_t rankSize, const std::string &ip, uint32_t port)
        : _rankId(rankId), _rankSize(rankSize), _ip(ip), _port(port) {};
    ~XSock(void);
    int Broadcast(void *buf, uint32_t size);
    int AllGather(void *buf, uint32_t size, void *allBuf);

private:
    int InitServer(void);
    int InitClient(void);
    int Init(void);
    int Recv(int fd, void *buf, uint32_t size);
    int Send(int fd, void *buf, uint32_t size);
    uint32_t _rankId;
    uint32_t _rankSize;
    std::string _ip;
    uint32_t _port;
    int _fd = -1;
    std::vector<int> _clientFds;
    bool _inited = false;
};

#endif