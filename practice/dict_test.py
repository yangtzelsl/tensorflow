import json


def generate_config():
    props = {}

    # 第一层
    source_props = {}
    process_props = {}
    sink_props = {}

    # 第二层
    source_with_props = {}
    source_with_props["connector"] = "kafka"
    source_with_props["topic"] = "instagram"
    source_with_props["properties.bootstrap.servers"] = "172.16.7.116:9092"
    source_with_props["properties.group.id"] = "test"
    source_with_props["format"] = "json"
    source_with_props["properties.security.protocol"] = "SASL_PLAINTEXT"
    source_with_props["properties.sasl.mechanism"] = "PLAIN"
    source_with_props[
        "properties.sasl.jaas.config"] = "org.apache.kafka.common.security.plain.PlainLoginModule required username=\"ptreader\" password=\"pt30@123\";"
    source_with_props["scan.startup.mode"] = "earliest-offset"

    source_props["sql"] = "select * from source"
    source_props["table_name"] = "instagram"
    source_props["schema"] = "checkLocation string,message string,type string"
    source_props["with"] = source_with_props

    process_props["sql"] = "select * from process"

    sink_with_props = {}
    sink_with_props["connector"] = "elasticsearch-7"
    sink_with_props["hosts"] = "http://172.16.7.52:19200"
    sink_with_props["index"] = "lzy_test"
    sink_props["sql"] = "select * from sink"
    sink_props["table_name"] = "myUserTable"
    sink_props["schema"] = "message string"
    sink_props["with"] = sink_with_props

    props["source"] = source_props
    props["process"] = process_props
    props["sink"] = sink_props
    props["parallelism"] = 4

    return props


def save_properties(properties, path):
    """
    保存配置, properties文件格式
    :param properties 配置内容
    :param path 配置文件路径
    """
    # 保存配置到本地文件
    with open(path, 'w', encoding='utf-8') as props:
        for k, v in properties.items():
            result = str(k) + "=" + str(v) + "\n"
            props.write(result)

    return 'success'


def save_conf(properties, path):
    """
    保存配置
    :param properties 配置内容
    :param path 配置文件路径
    """
    # 保存配置到本地文件
    with open(path, 'w', encoding='utf-8') as props:
        props.write(str(properties))

    return 'success'


if __name__ == "__main__":
    prop = generate_config()
    print(prop)

    result = save_conf(prop, "/props.properties")

    print(result)
