#
# Copyright 2011-2015 NVIDIA Corporation. All rights reserved
#
INCLUDES=/homelocal/fmendeslocal/GPU-Characterization

CC = hipcc

OBJ = o

NAME = eventPerformance_Mult_Int_dp

# export LD_LIBRARY_PATH=/opt/cuda/lib:/opt/cuda/lib64:$LD_LIBRARY_PATH
LIBS=

all: $(NAME)

hip: hipify $(NAME)

hipify:
	hipify-perl $(NAME).cu > $(NAME).cpp

$(NAME): $(NAME).$(OBJ)
	$(CC) -o $@ $(NAME).$(OBJ) $(LIBS)

$(NAME).$(OBJ): $(NAME).cpp
	$(CC) -c -I$(INCLUDES) $<

clean:
	rm -f $(NAME) $(NAME).$(OBJ)

