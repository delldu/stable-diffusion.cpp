#/************************************************************************************
#***
#***	Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
#***
#***	File Author: Dell, Thu 25 Apr 2024 03:15:39 PM CST
#***
#************************************************************************************/
#

PKGCONFNAME=stable-diffusion.pc

# Installation related variables and target
PREFIX?=/usr/local
LIBRARY_PATH?=lib
PKGCONF_PATH?=pkgconfig
INCLUDE_PATH?=include/stable-diffusion
INSTALL_INCLUDE_PATH= $(PREFIX)/$(INCLUDE_PATH)
INSTALL_LIBRARY_PATH= $(PREFIX)/$(LIBRARY_PATH)
INSTALL_PKGCONF_PATH= $(INSTALL_LIBRARY_PATH)/$(PKGCONF_PATH)

all: install

install:
	sudo mkdir -p ${INSTALL_INCLUDE_PATH}
	sudo cp ../stable-diffusion.h ${INSTALL_INCLUDE_PATH}
	sudo cp ../ggml/include/ggml/*.h ${INSTALL_INCLUDE_PATH}
	
	sudo cp ../build/libstable-diffusion.a ${INSTALL_LIBRARY_PATH}
	sudo cp ../build/ggml/src/libggml.a ${INSTALL_LIBRARY_PATH}

	sudo mkdir -p ${INSTALL_PKGCONF_PATH}
	sudo cp ${PKGCONFNAME} ${INSTALL_PKGCONF_PATH}

