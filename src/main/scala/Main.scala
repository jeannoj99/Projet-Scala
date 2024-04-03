import org.apache.spark.sql.{DataFrame,SparkSession}
import org.apache.spark.SparkConf

object Main extends App {
  val conf: SparkConf = new SparkConf()
  conf.set("spark.driver.memory","1G")
    .set("spark.testing.memory", "2147480000")

  val sparkSession = SparkSession
    .builder()
    .master("local[1]")
    .config(conf)
    .enableHiveSupport()
    .getOrCreate()

  val inputDF = sparkSession.sql("SELECT 'A' ")
    inputDF.show()
}

