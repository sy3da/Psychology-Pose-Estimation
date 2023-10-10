import os
import sys
import subprocess
import platform
import re

nb = sys.argv[1]

def write_brd_id(nb):

    # Find which os this script is running on
    os = platform.system()

    # By default use gpzvrc.exe (for windows)
    gpz_vrc = 'gpzvrc.exe'

    if(os == 'Windows'): # If on Windows
        gpz_vrc = 'gpzvrc.exe'
    elif(os == 'Darwin'): # If on MacOS
        gpz_vrc = './gpzvrc_mac'
        # Set permission of gpzvrc_mac, not executable by default
        chmod_test = subprocess.Popen(['chmod', '777', 'gpzvrc_mac'], shell=True, stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        chmod_test.wait()

    print("Current OS is {} and the application for this OS is {}".format(os,gpz_vrc))

    # List out all file in board directory (/mnt/boot)
    output_file = subprocess.Popen([gpz_vrc, 'run', 'ls /mnt/boot'], stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()[0]

    # Extract current sd image version
    sd_image = ''.join([n for n in output_file if n.isdigit()])
    print("The current sd image is: %s" %sd_image)

    # Find gpz.cfg file in directory /mnt/boot
    # If gpz.cfg exist
    if (output_file.find('gpz.cfg') != -1):
        print("gpz.cfg file exist!")

        # Look at the content of gpz.cfg file
        output_brd_id = subprocess.Popen([gpz_vrc, 'run', 'cat /mnt/boot/gpz.cfg'], stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()[0]

        # If BRD_ID is present
        if (output_brd_id.find('BRD_ID=') != -1):
            print("BRD_ID is present")
            m = re.search('BRD_ID=(\d+)', output_brd_id, re.IGNORECASE)
            print("The current BRD_ID is: %s" %m.group(1))
        else:
            print("BRD_ID is not present")

        print("Writing new BRD_ID=%s" %nb)
        # Find and Delete old BRD_ID
        sed_res = subprocess.Popen([gpz_vrc, 'run', 'sed -i \'/BRD_ID=/d\' /mnt/boot/gpz.cfg'],stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        # Wait for subprocess to finish
        sed_res.wait()

        # Write new BRD_ID
        echo_res = subprocess.Popen([gpz_vrc, 'run', 'echo BRD_ID=' + nb + ' >> /mnt/boot/gpz.cfg'], stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        # Wait for subprocess to finish
        echo_res.wait()
        print("Finished writing BRD_ID=%s" %nb)

    else:
        print("gpz.cfg file doesn't exist!")
        print("Creating gpz.cfg file and writing BRD_ID in gpz.cfg")
        # Creat gpz.cfg file
        touch_res = subprocess.Popen([gpz_vrc, 'run', 'touch /mnt/boot/gpz.cfg'], stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        # Wait for subprocess to finish
        touch_res.wait()

        # Write new BRD_ID
        write_res = subprocess.Popen([gpz_vrc, 'run', 'echo BRD_ID=' + nb +' > /mnt/boot/gpz.cfg'], stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        # Wait for subprocess to finish
        write_res.wait()
        print("Finished writing BRD_ID=%s" %nb)

    print("Rebooting board")
    # Reboot the board
    reboot_res = subprocess.Popen([gpz_vrc, 'run', 'reboot'], stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    reboot_res.wait()

if __name__ == '__main__':
    write_brd_id(nb)