from pyspark import SparkContext
import os
# os.environ ['JAVA_HOME'] = '/java/jdk1.8'
sc = SparkContext( 'local', 'test')

logFile = "file:///spark/spark/README.md"
logData = sc.textFile(logFile, 2).cache()
numAs = logData.filter(lambda line: 'a' in line).count()
numBs = logData.filter(lambda line: 'b' in line).count()
print('Lines with a: %s, Lines with b: %s' % (numAs, numBs))
