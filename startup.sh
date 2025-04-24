- name: Create startup script
  run: |
    echo "#!/bin/bash" > startup.sh
    echo "cd /home/site/wwwroot" >> startup.sh
    echo "gunicorn --bind=0.0.0.0 --timeout 600 api:app" >> startup.sh
    chmod +x startup.sh