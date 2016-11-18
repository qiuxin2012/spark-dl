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

package com.intel.analytics.sparkdl.dataset

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.File
import scopt.OptionParser

object Cifar10Test2 {
  case class Cifar10LocalParam(
    folder: String = "./",
    net: String = "vgg"
  )

  private val parser = new OptionParser[Cifar10LocalParam]("Spark-DL Cifar10 Local Example") {
    head("Spark-DL Cifar10 Local Example")
    opt[String]('f', "folder")
      .text("where you put the Cifar10 data")
      .action((x, c) => c.copy(folder = x))
  }

  def main(args: Array[String]) {
    parser.parse(args, new Cifar10LocalParam()).map(param => {
//      val trainDataSource = new CifarDataSource(Paths.get(param.folder + "/train"), looped = true)
////      val validationDataSource = new CifarDataSource(Paths.get(param.folder + "/val"),
////        looped = false)
//      val arrayToImage = ArrayByteToRGBImage(1)
//      val toTensor = new RGBImageToTensor(batchSize = 100)
//      val bgr2yuv = BgrToYuv()
//      val normalizer = YuvImageNormalizer(trainDataSource -> arrayToImage -> toTensor -> bgr2yuv)
////      val validationNormalizer = YuvImageNormalizer(validationDataSource -> arrayToImage -> toTensor -> bgr2yuv)
////      val model = VggLike[Float](classNum = 10)
////      println(model)
//      val data = trainDataSource -> arrayToImage -> toTensor -> bgr2yuv -> normalizer
//
//      val tensor = Tensor[Float](50000, 3, 32, 32)
//
//      data.reset()
//      var i = 1
//      while (!data.finished()) {
//        val images = data.next()._1
//        val batch = images.size(1)
//        tensor.narrow(1, i, batch).copy(images)
//        i += batch
//      }

//      File.save(tensor, "ts.obj")
      val tensor = File.loadObj[Tensor[Float]]("ts.obj")

      val luaTensor = File.load[Tensor[Float]]("/home/xin/workspaces/cifar.torch/trainf.t7")

//      println(tensor.mean())
//      println(luaTensor.mean())
//
//      println(tensor.select(2, 1).mean())
//      println(tensor.select(2, 2).mean())
//      println(tensor.select(2, 3).mean())
//      println(luaTensor.select(2, 1).mean())
//      println(luaTensor.select(2, 2).mean())
//      println(luaTensor.select(2, 3).mean())
//
//
//      println(tensor.sum())
//      println(luaTensor.sum())
//      println(tensor.select(2, 1).sum())
//      println(tensor.select(2, 2).sum())
//      println(tensor.select(2, 3).sum())
//      println(luaTensor.select(2, 1).sum())
//      println(luaTensor.select(2, 2).sum())
//      println(luaTensor.select(2, 3).sum())


      println(tensor.select(2, 2).sum())
      println(luaTensor.select(2, 2).sum())

      val i1 = Array(1, 1, 1)
      val i2 = Array(1, 1, 2)
      val i3 = Array(1, 1, 3)
      val j1 = Array(2, 1, 1)

      var x = 0
      var j = 1
      while (j <= 50000) {
        val a = tensor.select(1, j)
        val amean1 = a(1).mean()
        var i = 1
        while (i <= 50000) {
          val b = luaTensor.select(1, i)
          if (a(i1) / b(i1) < 1.01 && a(i1) / b(i1) > 0.99 && a(i2) / b(i2) < 1.01 && a(i2) / b(i2) > 0.99
          && a(i3) / b(i3) < 1.01 && a(i3) / b(i3) > 0.99 && amean1 / b(1).mean() < 1.01 && amean1 / b(1).mean() > 0.99 ) {
            x += 1
            println(i, j, x)
            if (a(2).mean() / b(2).mean > 1.01 || a(2).mean() / b(2).mean() < 0.99) {
              view(a, b)
            }
          }
          i += 1
        }
        j += 1
      }

      println()

    })
  }

  def view(a: Tensor[Float], b: Tensor[Float]): Unit = {
    println(a(2).mean())
    println(b(2).mean())
    println("aaaaaaaaaa")

  }
}
