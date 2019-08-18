import org.apache.spark.{SparkConf, SparkContext}

object SparkCode {

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("App").setMaster("local")
    val sparkContext = new SparkContext(sparkConf)

    val textFile = sparkContext.textFile(path = "src/resources/spark/sample.log",minPartitions = 2).cache()
    val processedTextFile = textFile.map(line => line.split(" ").length).reduce(Math.max)

    print(processedTextFile)
  }
}
