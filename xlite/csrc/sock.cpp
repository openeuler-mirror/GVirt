/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */

#include <cstdio>
#include <iostream>
#include <cstring>
#include <cerrno>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/in.h>
#include "sock.h"

#define ERRNO_NEED_RETYR(errno) ((errno) == EAGAIN || (errno) == EINTR || (errno) == EWOULDBLOCK)

XSock::~XSock(void)
{
    if (_fd >= 0) {
        close(_fd);
    }

    if (_clientFds.empty()) {
        return;
    }

    for (uint32_t i = 1; i < _rankSize; i++) {
        if (_clientFds[i] >= 0) {
            close(_clientFds[i]);
        }
    }
}

int XSock::InitServer(void)
{
    _clientFds.resize(_rankSize, -1);
    _fd = socket(AF_INET, SOCK_STREAM, 0);
    if (_fd < 0) {
        std::cerr << __func__ << ": rank" << _rankId << " server socket create failed: " <<
            strerror(errno) << std::endl;
        return -errno;
    }

    int reuse = 1;
    if (setsockopt(_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        std::cerr << __func__ << ": rank" << _rankId << " failed to set socket option: " <<
            strerror(errno) << std::endl;
        return -errno;
    }

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(_ip.c_str());
    addr.sin_port = htons(_port);
    if (bind(_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(sockaddr)) < 0) {
        std::cerr << __func__ << ": rank" << _rankId << " failed to bind server socket: " <<
            strerror(errno) << std::endl;
        return -errno;
    }

    if (listen(_fd, _rankSize - 1) == -1) {
        std::cerr << __func__ << ": rank" << _rankId << " failed to listen on server socket: " <<
            strerror(errno) << std::endl;
        return -errno;
    }

    for (uint32_t i = 1; i < _rankSize; i++) {
        int clientFd = -1;
        int ret = 0;
        do {
            sockaddr_in clientAddr{};
            socklen_t clientAddrLen = sizeof(clientAddr);
            clientFd = accept(_fd, reinterpret_cast<sockaddr*>(&clientAddr), &clientAddrLen);
            ret = errno;
        } while (clientFd < 0 && ERRNO_NEED_RETYR(ret));

        if (clientFd < 0) {
            std::cerr << __func__ << ": rank" << _rankId << " accept failed: " << strerror(errno) << std::endl;
            return -ret;
        }

        uint32_t rank = 0;
        ret = Recv(clientFd, &rank, sizeof(rank));
        if (ret < 0) {
            std::cerr << __func__ << ": rank" << _rankId << " recv rank id failed" << std::endl;
            close(clientFd);
            return ret;
        }

        if (rank > _rankSize || rank <= 0 || _clientFds[rank] >= 0) {
            std::cerr << __func__ << ": rank" << _rankId << " recv invalid rank id" << std::endl;
            close(clientFd);
            return -EFAULT;
        }

        _clientFds[rank] = clientFd;
    }

    _inited = true;
    return 0;
}

int XSock::InitClient(void)
{
    _fd = socket(AF_INET, SOCK_STREAM, 0);
    if (_fd < 0) {
        std::cerr << __func__ << ": rank" << _rankId << " client socket create failed: " <<
            strerror(errno) << std::endl;
        return -errno;
    }

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(_ip.c_str());
    addr.sin_port = htons(_port);
    int retry = 0;
    bool success = false;
    while (retry < 10) {
        if (connect(_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(sockaddr)) < 0) {
            if (errno == ECONNREFUSED) {
                retry++;
                sleep(1);
                continue;
            }
            if (errno == EINTR) {
                continue;
            }
            std::cerr << __func__ << ": rank" << _rankId << " client connect failed: " << strerror(errno) << std::endl;
            break;
        }
        success = true;
        break;
    }
    if (!success) {
        std::cerr << __func__ << ": rank" << _rankId << " client connect timeout" << std::endl;
        return -EFAULT;
    }

    int ret = Send(_fd, &_rankId, sizeof(_rankId));
    if (ret < 0) {
        std::cerr << __func__ << ": rank" << _rankId << " client send rank id failed" << std::endl;
        return ret;
    }

    _inited = true;
    return 0;
}

int XSock::Init(void)
{
    if (_rankId != 0) {
        return InitClient();
    } else {
        return InitServer();
    }
}

int XSock::Recv(int fd, void *buf, uint32_t size)
{
    int ret = 0;
    do {
        ret = recv(fd, buf, size, 0);
    } while (ret < 0 && ERRNO_NEED_RETYR(errno));

    if (ret == 0) {
        std::cerr << __func__ << ": rank" << _rankId << " recv failed, connect reset by peer" << std::endl;
        return -ECONNRESET;
    } else if (ret < 0) {
        std::cerr << __func__ << ": rank" << _rankId << " recv failed: " << strerror(errno) << std::endl;
        return -errno;
    } else if ((uint32_t)ret != size) {
        std::cerr << __func__ << ": rank" << _rankId << " recv failed: size not match" << std::endl;
        return -EFAULT;
    }
    return 0;
}

int XSock::Send(int fd, void *buf, uint32_t size)
{
    int ret = 0;
    do {
        ret = send(fd, buf, size, 0);
    } while (ret < 0 && ERRNO_NEED_RETYR(errno));

    if (ret == 0) {
        std::cerr << __func__ << ": rank" << _rankId << " send failed, connect reset by peer" << std::endl;
        return -ECONNRESET;
    } else if (ret < 0) {
        std::cerr << __func__ << ": rank" << _rankId << " send failed: " << strerror(errno) << std::endl;
        return -errno;
    } else if ((uint32_t)ret != size) {
        std::cerr << __func__ << ": rank" << _rankId << " send failed: size not match" << std::endl;
        return -EFAULT;
    }
    return 0;
}

int XSock::Broadcast(void *buf, uint32_t size)
{
    if (!_inited && Init()) {
        return -EFAULT;
    }
    if (_rankId != 0) {
        return Recv(_fd, buf, size);
    } else {
        for (uint32_t rank = 1; rank < _rankSize; rank++) {
            if (Send(_clientFds[rank], buf, size) < 0) {
                return -EFAULT;
            }
        }
    }
    return 0;
}

int XSock::AllGather(void *buf, uint32_t size, void *allBuf)
{
    if (!_inited && Init()) {
        return -EFAULT;
    }
    if (_rankId != 0) {
        Send(_fd, buf, size);
        Recv(_fd, allBuf, size * _rankSize);
    } else {
        for (uint32_t rank = 1; rank < _rankSize; rank++) {
            Recv(_clientFds[rank], (void *)((uint64_t)allBuf + size * rank), size);
        }
        for (uint32_t rank = 1; rank < _rankSize; rank++) {
            Send(_clientFds[rank], allBuf, size * _rankSize);
        }
    }
    return 0;
}