from hdfs.client import Client, InsecureClient

# The address and the base port where the dfs namenode web ui will listen on.
# client = Client("http://172.16.7.59:9870",root='/user/flink')
client = InsecureClient("http://172.16.7.59:9870",user='flink')

# list_file = client.list('/user')
# print(list_file)

local_path = '/test.properties'
hdfs_path = '/user/flink/hdfs_py'


def read_hdfs(hdfs_path):
    with client.read(hdfs_path, chunk_size=1280) as reader:
        for chunk in reader:
            print(chunk.decode().replace(' ', '\n '))


def write_hdfs(local_path, hdfs_path):
    """
    将本地文件写入到 HDFS
    :param local_path: 本地文件路径
    :param hdfs_path: HDFS文件路径
    :return:
    """
    with open(local_path) as opener, client.write(hdfs_path) as writer:
        for line in opener:
            writer.write(bytes(line, encoding='utf-8'))


# 读取hdfs文件内容,将每行存入数组返回
def read_hdfs_file(client, filename):
    # with client.read('samples.csv', encoding='utf-8', delimiter='\n') as reader:
    #  for line in reader:
    # pass
    lines = []
    with client.read(filename, encoding='utf-8', delimiter='\n') as reader:
        for line in reader:
            # pass
            # print line.strip()
            lines.append(line.strip())
    return lines


# 创建目录
def mkdirs(client, hdfs_path):
    client.makedirs(hdfs_path)


# 删除hdfs文件
def delete_hdfs_file(client, hdfs_path):
    client.delete(hdfs_path)


# 上传文件到hdfs
def put_to_hdfs(client, local_path, hdfs_path):
    client.upload(hdfs_path, local_path, cleanup=True)


# 从hdfs获取文件到本地
def get_from_hdfs(client, hdfs_path, local_path):
    client.download(hdfs_path, local_path, overwrite=False)


# 追加数据到hdfs文件
def append_to_hdfs(client, hdfs_path, data):
    client.write(hdfs_path, data, overwrite=False, append=True, encoding='utf-8')


# 覆盖数据写到hdfs文件
def write_to_hdfs(client, hdfs_path, data):
    client.write(hdfs_path, data, overwrite=True, append=False, encoding='utf-8')


# 移动或者修改文件
def move_or_rename(client, hdfs_src_path, hdfs_dst_path):
    client.rename(hdfs_src_path, hdfs_dst_path)


# 返回目录下的文件
def list(client, hdfs_path):
    return client.list(hdfs_path, status=False)


# client = Client(url, root=None, proxy=None, timeout=None, session=None)
# client = Client("http://hadoop:50070")
# client = Client("http://120.78.186.82:50070/", root="/", timeout=10000, session=False)
# client = InsecureClient("http://120.78.186.82:50070", user='ann');

# move_or_rename(client,'/input/2.csv', '/input/emp.csv')
# read_hdfs_file(client,'/input/emp.csv')
put_to_hdfs(client, local_path, hdfs_path)
# append_to_hdfs(client,'/input/emp.csv','我爱你'+'\n')
# write_to_hdfs(client, '/emp.csv', "sadfafdadsf")
# read_hdfs_file(client,'/input/emp.csv')
# move_or_rename(client,'/input/emp.csv', '/input/2.csv')
# mkdirs(client,'/input/python')
# print(list(client, '/'))
# chown(client,'/input/1.csv', 'root')


# write_hdfs(local_path,hdfs_path)