from pyspark import SparkContext
from pyspark.sql import SparkSession

def load_data(from_file: bool, sc: SparkContext):
    if from_file:
        return load_data_from_file(sc)
    return load_data_from_in_memory(sc)


def load_data_from_file(sc: SparkContext):
    user_visits_data = sc.textFile("data/spark/user_visits.txt").map(lambda line: line.split(','))
    user_names_data = sc.textFile("data/spark/user_names.txt").map(lambda line: line.split(','))
    return user_visits_data, user_names_data

def load_data_from_in_memory(sc: SparkContext):
    # [user_id, visits]
    user_visits = [
        (1, 10),
        (2, 27),
        (3, 2),
        (4, 5),
        (5, 88),
        (6, 1),
        (7, 5)
    ]
    # [userid, name]
    user_names = [
        (1, "Andrew"),
        (2, "Chris"),
        (3, "John"),
        (4, "Bob"),
        (6, "Ryan"),
        (7, "Mali"),
        (8, "Tony"),
    ]

    return sc.parallelize(user_visits), sc.parallelize(user_names)

if __name__ == '__main__':
    ss: SparkSession = SparkSession.builder.master('local').appName('join').getOrCreate()
    sc: SparkContext = ss.sparkContext

    # user_visits_rdd, user_names_rdd = load_data(sc)
    user_visits_rdd, user_names_rdd = load_data(from_file=True, sc=sc)

    # print(f"user_visits_rdd: {user_visits_rdd.take(num=5)}")
    # print(f"user_names_rdd: {user_names_rdd.take(num=5)}")

    inner_joined_rdd = user_names_rdd.join(user_visits_rdd).sortByKey()
    print(f"inner_joined_rdd: {inner_joined_rdd.collect()}")

    left_outer_joined_rdd = user_names_rdd.leftOuterJoin(user_visits_rdd).sortByKey()
    print(f"left_outer_joined_rdd: {left_outer_joined_rdd.collect()}")

    right_outer_joined_rdd = user_names_rdd.rightOuterJoin(user_visits_rdd).sortByKey()
    print(f"right_outer_joined_rdd: {right_outer_joined_rdd.collect()}")

    full_outer_joined_rdd = user_names_rdd.fullOuterJoin(user_visits_rdd).sortByKey()
    print(f"full_outer_joined_rdd: {full_outer_joined_rdd.collect()}")