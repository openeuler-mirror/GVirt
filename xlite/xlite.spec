Name:           xlite
Version:        0.1.0rc4
Release:        1%{?dist}
Summary:        A lightweight, effective and easy-to-extend inference runtime

License:        MulanPSL-2.0
Source0:        %{name}-%{version}.tar.gz

# Disable automatic dependency generation; equivalent to `rpm --nodeps` for the generated package.
AutoReq:        no

BuildRequires:  gcc-c++, cmake >= 3.16, make
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
export SETUPTOOLS_SCM_PRETEND_VERSION=%{version}
%py3_build

%install
rm -rf %{buildroot}
%py3_install
rm -rf %{buildroot}/%{python3_sitearch}/csrc
# Drop Python bytecode caches so they do not need per-path %exclude entries.
find %{buildroot}/%{python3_sitearch}/xlite -type d -name __pycache__ -prune -exec rm -rf {} +
find %{buildroot}/%{python3_sitearch}/xlite -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
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
%exclude %{python3_sitearch}/**/__pycache__
%exclude %{python3_sitearch}/**/*.pyc
%exclude %{python3_sitearch}/**/*.pyo

%changelog
* Thu Oct 16 2025 wangxiaoran <wangxiaoran11@huawei.com> - 1.0-1
- Adapt rpm spec for building.
* Mon May 19 2025 lulina <lina.lulina@huawei.com> - 1.0-1
- Initial package for xlite.
