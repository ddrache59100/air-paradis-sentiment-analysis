- name: Create build script
  run: |
    echo "#!/bin/bash" > build.sh
    echo "echo 'Installing dependencies...'" >> build.sh
    echo "pip install -r requirements.txt" >> build.sh
    chmod +x build.sh