docker stop hive_c
docker rm hive_c
docker run -it --name hive_c -v $(pwd)/../../:/opt/hive/ hive /bin/bash
