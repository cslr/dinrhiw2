#!/bin/bash

echo -e "\n"
echo -e "Copying template cpp files to library...\n"

fgrep -dskip "#include" src/* | grep "h:#include" | grep cpp\" | perl -e 'while(<>){ m/[^"]*"([^"]*)".*/; $filename = "`find src/* | grep -e cpp\$ | grep $1`"; print "$1\n"; `cp -f $filename lib/` }'
fgrep -dskip "#include" src/math/* | grep "h:#include" | grep cpp\" | perl -e 'while(<>){ m/[^"]*"([^"]*)".*/; $filename = "`find src/* | grep -e cpp\$ | grep $1`"; print "$1\n"; `cp -f $filename lib/` }'
fgrep -dskip "#include" src/crypto/* | grep "h:#include" | grep cpp\" | perl -e 'while(<>){ m/[^"]*"([^"]*)".*/; $filename = "`find src/* | grep -e cpp\$ | grep $1`"; print "$1\n"; `cp -f $filename lib/` }'
fgrep -dskip "#include" src/neuralnetwork/* | grep "h:#include" | grep cpp\" | perl -e 'while(<>){ m/[^"]*"([^"]*)".*/; $filename = "`find src/* | grep -e cpp\$ | grep $1`"; print "$1\n"; `cp -f $filename lib/` }'

echo -e "\n"
echo -e "DINRHIW library has been build.\n"
echo -e "\n"
