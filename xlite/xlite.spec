Name:           xlite
Version:        1.0
Release:        1%{?dist}
Summary:        A lightweight, effective and easy-to-extend inference runtime

License:        MulanPSL2
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  gcc-c++
BuildRequires:  cmake

%description
Xlite is A lightweight, effective and easy-to-extend inference runtime.

%define ASCEND_INSTALL_PATH /usr/local/Ascend/ascend-toolkit/latest

%pre
if [ ! -d %{ASCEND_INSTALL_PATH} ]; then
    echo "Please install Ascend-cann." >&2
fi

%prep
%autosetup
rm -rf build out

%build
%py3_build

%install
rm -rf $RPM_BUILD_ROOT
%py3_install

%post
/sbin/ldconfig

%postun
if [ "$1" = "0" ] || [ "$1" = "remove" ] || [ "$1" = "purge" ]; then
    rm -rf %{python3_sitelib}/xlite*
    /sbin/ldconfig
fi

%files
%doc README.md

%{python3_sitelib}/xlite*

%changelog
* Mon May 19 2025 lulina <lina.lulina@huawei.com> - 1.0-1
- Initial package for xlite.