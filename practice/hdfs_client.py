from hdfs.client import Client

client = Client("http://172.16.7.59:9870")

list_file = client.list('/user')
print(list_file)

local_path = '/test.properties'
hdfs_path = '/opt/'


def read_hdfs(hdfs_path):
    with client.read(hdfs_path, chunk_size=1280) as reader:
        for chunk in reader:
            print(chunk.decode().replace(' ', '\n '))


def write_hdfs(local_path, hdfs_path):
    with open(local_path) as opener, client.write(hdfs_path) as writer:
        for line in opener:
            writer.write(bytes(line, encoding='utf-8'))

write_hdfs(local_path,hdfs_path)