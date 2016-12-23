/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, Row, functions}

import scala.collection.mutable.ArrayBuffer

class DLClassifier(override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with DataParams{

  var tensorBuffer: Tensor[Float] = null

  def this() = this(Identifiable.randomUID("DLClassifier"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: DataFrame): DataFrame = {
    require(null != $(modelTrain), "model for predict must not be null")
    require(null != $(batchSize), "batchSize for predict must not be null")
    DLClassifier.transform($(batchSize), $(modelTrain), $(inputCol), $(outputCol), dataset)
  }

  override def transformSchema(schema: StructType): StructType = schema

  override def copy(extra: ParamMap): DLClassifier = {
    copyValues(new DLClassifier(uid), extra)
  }
}

object  DLClassifier {
  private[DLClassifier] def transform(
      batchSize: Array[Int],
      modelTrain: Module[Float],
      inputCol: String,
      outputCol: String,
      dataset: DataFrame): DataFrame = {
    val model = modelTrain.evaluate()

    val modelBroadCast = dataset.sqlContext.sparkContext.broadcast(model)

    val predictRdd = dataset.rdd.mapPartitions{ rows =>
      val result = new ArrayBuffer[Row]()
      val localModel = modelBroadCast.value
      val tensorBuffer = Tensor[Float](batchSize)
      val batches = rows.grouped(batchSize(0))

      while (batches.hasNext) {
        val batch = batches.next()

        var i = 1
        batch.foreach{ row =>
          tensorBuffer.select(1, i).copy(
            Tensor(Storage(row.getAs[DenseVector](inputCol).values.map(_.toFloat))))
          i += 1
        }
        val output = localModel.forward(tensorBuffer).toTensor[Float]
        val predict = if (output.dim == 2) {
          output.max(2)._2.squeeze().storage().array()
        } else if (output.dim == 1) {
          output.max(1)._2.squeeze().storage().array()
        } else {
          throw new IllegalArgumentException
        }

        i = 0
        batch.foreach{ row =>
          result.append(Row.fromSeq(row.toSeq ++ Array[Int](predict(i).toInt)))
          i += 1
        }
      }
      result.toIterator
    }
    val predictSchema = dataset.schema.add(outputCol, IntegerType)
    dataset.sqlContext.createDataFrame(predictRdd, predictSchema)
  }
}

trait DataParams extends Params {
  final val modelTrain = new Param[Module[Float]](this, "module factory", "network model")
  final val batchSize = new Param[Array[Int]](this, "batch size", "batch size for input")

  final def getModel = $(modelTrain)
  final def getBatchSize = $(batchSize)
}
