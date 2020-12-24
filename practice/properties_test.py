import json

properties_json = {
    "config": {
        "source": [
            {
                "with": {
                    "connector": "kafka",
                    "topic": "instagram",
                    "bootstrap.servers": "172.16.7.10:9092,172.16.7.11:9092,172.16.7.12:9092",
                    "group.id": "flink_facebook_data_quality",
                    "format": "json"
                },
                "id": "nodeA",
                "table_name": "instagram",
                "schema": "checkLocation string, message string, type string"
            }
        ],
        "process": [
            {
                "id": "nodeB",
                "sql": "slect xx",
                "table_name": "aaa"
            },
            {
                "id": "nodeB1",
                "sql": "slect xx",
                "table_name": "bbb"
            },
            {
                "id": "nodeB2",
                "sql": "slect xx",
                "table_name": "ccc"
            }
        ],
        "sink": [
            {
                "with": {
                    "format": "json",
                    "connector": "elasticsearch-7",
                    "hosts": "http://172.16.7.52:19200",
                    "index": "lzy_test"
                },
                "id": "nodeC",
                "table_name": "myUserTable",
                "schema": "message string"
            }
        ],
        "lines": {
            "seq": [
                [
                    "nodeA",
                    "nodeB"
                ],
                [
                    "nodeB",
                    "nodeB1"
                ],
                [
                    "nodeB",
                    "nodeB2"
                ],
                [
                    "nodeB1",
                    "nodeC"
                ],
                [
                    "nodeB2",
                    "nodeC"
                ]
            ]
        },
        "parallelism": 2
    },
    "parallelism": 2
}


with open("config.properties", "w", encoding='utf-8') as fp:
    for key in properties_json.keys():
        print(json.dumps(properties_json[key]))
        print(json.dumps(properties_json[key])[1:-1])
        print(json.dumps(json.dumps(properties_json[key])))
        print(json.dumps(json.dumps(properties_json[key]))[1:-1])
        fp.write(key + "=" + json.dumps(json.dumps(properties_json[key]), indent=4, ensure_ascii=False)[1:-1] + "\n")
        # json.dumps(dict) 将dict转化成str格式
        # json.loads(str) 将str转化成dict格式
        # json.dump(dict, fp) 将dict转成str然后存入文件中
        # json.load(fp) 读一个json的文件转成str


def jprops_fun():
    import jprops
    with open("config.properties", "w", encoding='utf-8') as fp:
        jprops.store_properties(fp, properties_json)


def jproperties_fun():
    from jproperties import Properties
    p = Properties()

    # Or with default values:
    p2 = Properties({"foo": "bar", "aaa": "bbb"})
    p3 = Properties(properties_json)
    # print(p2["foo"])
    # p2["key"] = "value"
    # print(p3.keys())
    for key in p3.keys():
        print(key)
        with open("out.properties", "w") as f:
            # f.write(key+"="+json.dumps(p3[key])+"\n")
            f.write(json.dumps(p3[key]))
