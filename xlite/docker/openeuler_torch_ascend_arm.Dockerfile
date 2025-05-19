FROM hub.oepkgs.net/openeuler/openeuler:22.03-lts-sp4

####################### os #######################
RUN echo "[OS]" > /etc/yum.repos.d/openEuler.repo && \
    echo "name=OS" >> /etc/yum.repos.d/openEuler.repo && \
    echo "baseurl=https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/OS/\$basearch/" >> /etc/yum.repos.d/openEuler.repo && \
    echo "enabled=1" >> /etc/yum.repos.d/openEuler.repo && \
    echo "gpgcheck=1" >> /etc/yum.repos.d/openEuler.repo && \
    echo "gpgkey=https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/OS/\$basearch/RPM-GPG-KEY-openEuler" >> /etc/yum.repos.d/openEuler.repo && \
    echo "" >> /etc/yum.repos.d/openEuler.repo && \
    echo "[everything]" >> /etc/yum.repos.d/openEuler.repo && \
    echo "name=everything" >> /etc/yum.repos.d/openEuler.repo && \
    echo "baseurl=https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/everything/\$basearch/" >> /etc/yum.repos.d/openEuler.repo && \
    echo "enabled=1" >> /etc/yum.repos.d/openEuler.repo && \
    echo "gpgcheck=1" >> /etc/yum.repos.d/openEuler.repo && \
    echo "gpgkey=https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/everything/\$basearch/RPM-GPG-KEY-openEuler" >> /etc/yum.repos.d/openEuler.repo && \
    echo "" >> /etc/yum.repos.d/openEuler.repo && \
    echo "[update]" >> /etc/yum.repos.d/openEuler.repo && \
    echo "name=update" >> /etc/yum.repos.d/openEuler.repo && \
    echo "baseurl=https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/update/\$basearch/" >> /etc/yum.repos.d/openEuler.repo && \
    echo "enabled=1" >> /etc/yum.repos.d/openEuler.repo && \
    echo "gpgcheck=1" >> /etc/yum.repos.d/openEuler.repo && \
    echo "gpgkey=https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/update/\$basearch/RPM-GPG-KEY-openEuler" >> /etc/yum.repos.d/openEuler.repo
RUN echo "sslverify=false" >> /etc/yum.conf
RUN yum clean all && \
    yum makecache && \
    yum install -y \
        kmod \
        sudo \
        wget \
        curl \
        cmake \
        make \
        git \
        vim \
        gcc \
        rpm-build && \
    yum clean all

####################### python #######################
RUN yum install -y llvm-toolset-17 gperftools gperftools-devel pciutils xz gzip zip tree zlib-devel openssl-devel  libffi-devel xz-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel dpkg-devel libdb-devel gdbm-devel && yum clean all
ENV LD_PRELOAD="$LD_PRELOAD:/usr/lib64/libtcmalloc.so"

ENV CC=/opt/openEuler/llvm-toolset-17/root/usr/bin/clang
ENV CXX=/opt/openEuler/llvm-toolset-17/root/usr/bin/clang++
    	
RUN source /opt/openEuler/llvm-toolset-17/enable && clang -v && \
    wget --no-check-certificate -O Python-3.11.11.tgz \
    https://www.python.org/ftp/python/3.11.11/Python-3.11.11.tgz && \
    tar -zxvf Python-3.11.11.tgz -C /root/ && \
	rm Python-3.11.11.tgz && \
	cd /root/Python-3.11.11/ && \
	./configure --prefix=/root/Python-3.11 --with-lto --enable-optimizations && \
	make -j && make install  && \
	ln -s /root/Python-3.11/bin/pip3 /root/Python-3.11/bin/pip && \
	ln -s /root/Python-3.11/bin/python3 /root/Python-3.11/bin/python && \
	rm -rf /root/Python-3.11.11/

ENV PATH=/root/Python-3.11/bin:$PATH
ENV LD_LIBRARY_PATH=/root/Python-3.11/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/root/Python-3.11/lib/python3.11/site-packages:$PYTHONPATH
RUN pip config set global.index-url 'https://mirrors.huaweicloud.com/repository/pypi/simple' && \
    pip config set global.trusted-host 'mirrors.huaweicloud.com'

RUN pip install pybind11 && pip install pyyaml && pip install transformers && pip install decorator && pip install scipy && pip install attrs && pip install psutil && pip install wheel

####################### CANN #######################
WORKDIR /root
RUN echo "UserName=HwHiAiUser" >> /etc/ascend_install.info && \
    echo "UserGroup=HwHiAiUser" >> /etc/ascend_install.info && \
    echo "Firmware_Install_Type=full" >> /etc/ascend_install.info && \
    echo "Firmware_Install_Path_Param=/usr/local/Ascend" >> /etc/ascend_install.info && \
    echo "Driver_Install_Type=full" >> /etc/ascend_install.info && \
    echo "Driver_Install_Path_Param=/usr/local/Ascend" >> /etc/ascend_install.info && \
    echo "Driver_Install_For_All=no" >> /etc/ascend_install.info && \
    echo "Driver_Install_Mode=normal" >> /etc/ascend_install.info && \
    echo "Driver_Install_Status=complete" >> /etc/ascend_install.info
RUN curl -s -k "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.0/Ascend-cann-toolkit_8.0.0_linux-aarch64.run" -o Ascend-cann-toolkit.run && \
    chmod a+x *.run && \
    bash /root/Ascend-cann-toolkit.run --install -q && \
    rm /root/*.run
RUN curl -s -k "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.0/Ascend-cann-kernels-910b_8.0.0_linux-aarch64.run" -o Ascend-cann-kernels-910b.run && \
    chmod a+x *.run && \
    bash /root/Ascend-cann-kernels-910b.run --install -q && \
    rm /root/*.run
RUN echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /root/.bashrc
RUN echo "source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash" >> /root/.bashrc

####################### torch npu #######################
RUN pip install 'numpy<2.0.0' && pip install torchvision==0.16.0
RUN pip install torch==2.4.0 && pip install torch-npu==2.4.0