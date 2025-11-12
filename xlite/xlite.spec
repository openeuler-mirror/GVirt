Name:           xlite
Version:        1.0
Release:        1%{?dist}
Summary:        A lightweight, effective and easy-to-extend inference runtime

License:        MulanPSL2
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  gcc-c++, cmake, make
BuildRequires:  pciutils
BuildRequires:  zlib-devel
BuildRequires:  openssl-devel
BuildRequires:  libffi-devel
BuildRequires:  xz-devel, bzip2, bzip2-devel
BuildRequires:  readline-devel
BuildRequires:  sqlite, sqlite-devel
BuildRequires:  dpkg-devel
BuildRequires:  libdb-devel
BuildRequires:  gdbm-devel
BuildRequires:  python3-devel
BuildRequires:  python3-pip
BuildRequires:  python3-setuptools

%description
Xlite is A lightweight, effective and easy-to-extend inference runtime.

%define ASCEND_INSTALL_PATH /usr/local/Ascend/ascend-toolkit/latest

%global debug_package %{nil}

%pre
if [ ! -d %{ASCEND_INSTALL_PATH} ]; then
    echo "Please install Ascend-cann." >&2
fi
echo "Please install python packages in requirements.txt before building xlite"

%prep
%autosetup
rm -rf build out

%build
%py3_build

%install
rm -rf %{buildroot}
%py3_install
cp -r xlite/*.so %{buildroot}/%{python3_sitearch}/xlite
cp -r xlite/lib/*.so %{buildroot}/%{python3_sitearch}/xlite
cp -r tests %{buildroot}/%{python3_sitearch}/xlite

%post
/sbin/ldconfig

%postun
if [ "$1" = "0" ] || [ "$1" = "remove" ] || [ "$1" = "purge" ]; then
    rm -rf %{python3_sitearch}/xlite*
    /sbin/ldconfig
fi

%files
%doc README.md
%{python3_sitearch}/xlite/
%{python3_sitearch}/xlite*.egg-info/
%exclude %{python3_sitearch}/tests
%exclude %{python3_sitearch}/xlite/__pycache__
%exclude %{python3_sitearch}/xlite/tests/kernels/__pycache__
%exclude %{python3_sitearch}/xlite/tests/funcs/__pycache__
%exclude %{python3_sitearch}/xlite/tests/models/__pycache__
%exclude %{python3_sitearch}/xlite/tests/__pycache__
%exclude %{python3_sitearch}/xlite/tools/quantization/algorithms/__pycache__
%exclude %{python3_sitearch}/xlite/tools/quantization/__pycache__
%exclude %{python3_sitearch}/xlite/tools/__pycache__

%changelog
* Tue Oct 16 2025 wangxiaoran <wangxiaoran11@huawei.com> - 1.0-1
- Adapt rpm spec for building.
* Mon May 19 2025 lulina <lina.lulina@huawei.com> - 1.0-1
- Initial package for xlite.
