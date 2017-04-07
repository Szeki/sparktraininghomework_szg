package com.epam.training.spark.core

import java.time.LocalDate

import com.epam.training.spark.core.domain.Climate
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.DayOfMonth
import org.apache.spark.{SparkConf, SparkContext}

object Homework {
  val DELIMITER = ";"
  val RAW_BUDAPEST_DATA = "data/budapest_daily_1901-2010.csv"
  val OUTPUT_DUR = "output"

  def main(args: Array[String]): Unit = {
    val sparkConf: SparkConf = new SparkConf()
      .setAppName("EPAM BigData training Spark Core homework")
      .setIfMissing("spark.master", "local[2]")
      .setIfMissing("spark.sql.shuffle.partitions", "10")
    val sc = new SparkContext(sparkConf)

    processData(sc)

    sc.stop()

  }

  def processData(sc: SparkContext): Unit = {

    /**
      * Task 1
      * Read raw data from provided file, remove header, split rows by delimiter
      */
    val rawData: RDD[List[String]] = getRawDataWithoutHeader(sc, Homework.RAW_BUDAPEST_DATA)

    /**
      * Task 2
      * Find errors or missing values in the data
      */
    val errors: List[Int] = findErrors(rawData)
    println(errors)

    /**
      * Task 3
      * Map raw data to Climate type
      */
    val climateRdd: RDD[Climate] = mapToClimate(rawData)

    /**
      * Task 4
      * List average temperature for a given day in every year
      */
    val averageTemeperatureRdd: RDD[Double] = averageTemperature(climateRdd, 1, 2)

    /**
      * Task 5
      * Predict temperature based on mean temperature for every year including 1 day before and after
      * For the given month 1 and day 2 (2nd January) include days 1st January and 3rd January in the calculation
      */
    val predictedTemperature: Double = predictTemperature(climateRdd, 1, 2)
    println(s"Predicted temperature: $predictedTemperature")

  }

  def getRawDataWithoutHeader(sc: SparkContext, rawDataPath: String): RDD[List[String]] =
    sc.textFile(rawDataPath)
      .filter(line => !line.startsWith("#"))
      .map(line => line.split(";", -1).toList)

  def findErrors(rawData: RDD[List[String]]): List[Int] =
    rawData
      .map(list => list.map(getMissingDataIndicator(_)))
      .reduce((l1, l2) => (l1, l2).zipped map(_ + _))

  def mapToClimate(rawData: RDD[List[String]]): RDD[Climate] =
    rawData.map(list => Climate(list(0), list(1), list(2), list(3), list(4), list(5), list(6)))

  def averageTemperature(climateData: RDD[Climate], month: Int, dayOfMonth: Int): RDD[Double] =
    climateData
      .filter(c => isObservedDate(c.observationDate, month, dayOfMonth))
      .map(_.meanTemperature.value)

  def predictTemperature(climateData: RDD[Climate], month: Int, dayOfMonth: Int): Double = {
    val (sum, count) =
      climateData
        .filter(c => isObseredDateOrPrevNextDay(c.observationDate, month, dayOfMonth))
        .map(c => (c.meanTemperature.value, 1.0))
        .reduce((x, y) => (x._1 + y._1, x._2 + y._2))

    sum / count
  }

  private def getMissingDataIndicator(value: String) : Int =
    if(value == null || value.isEmpty) 1 else 0

  private def isObservedDate(date: LocalDate, month: Int, dayOfMonth: Int) : Boolean =
    date.getMonth.getValue == month && date.getDayOfMonth == dayOfMonth

  private def isObseredDateOrPrevNextDay(date: LocalDate, month: Int, dayOfMonth: Int) : Boolean =
    isObservedDate(date.minusDays(1), month, dayOfMonth) || isObservedDate(date, month, dayOfMonth) || isObservedDate(date.plusDays(1), month, dayOfMonth)
}


