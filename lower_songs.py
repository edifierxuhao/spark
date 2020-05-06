from pyspark.sql import SparkSession

if __name__ == '__main__':
    '''
    
    '''
    
    spark = SparkSession\
        .builder\
        .appName('LowerSongTitles')\
        .getOrGreat()
    
    log_of_songs = [
        'Despacito',
        'Nice for what',
        'No tears left to cry',
        'Despacito',
        'Havana',
        'In my feelings',
        'Nice for what',
        'despacito',
        'All the stars']
    
    distributed_song_log = spark.Context.parallelize(log_of_songs)
    
    print(distributed_song_log.map(lambda x: x.lower()).collect())
    
    spark.stop()
