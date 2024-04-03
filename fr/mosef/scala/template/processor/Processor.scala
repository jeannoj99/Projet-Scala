package fr.mosef.scala.template.processor

import org.apache.spark.sql.DataFrame

trait Processor {

  def process(inputDF: DataFrame) : DataFrame

}
