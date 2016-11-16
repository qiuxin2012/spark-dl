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

package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.T

import scala.reflect.ClassTag

class SpatialSubtractiveNormalization[T: ClassTag](
  val nInputPlane: Int = 1,
  var kernel: Tensor[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  if (kernel == null) kernel = Tensor.ones[T](9, 9)

  private val kdim = kernel.nDimension()
  require(kdim == 1 || kdim == 2, "averaging kernel must be 2D or 1D")
  require(kernel.size(1) % 2 != 0, "averaging kernel must have ODD dimensions")
  if (kdim == 2) {
    require(kernel.size(2) % 2 != 0, "averaging kernel must have ODD dimensions")
  }

  kernel.div(ev.times(kernel.sum(), ev.fromType[Int](nInputPlane)))

  val padH = math.floor(kernel.size(1).toFloat/2).toInt
  val padW = if (kdim == 2) {
    math.floor(kernel.size(2).toFloat/2).toInt
  } else {
    padH
  }

  // create convolutional mean extractor
  val meanestimator = new Sequential[Tensor[T], Tensor[T], T]()
  meanestimator.add(new SpatialZeroPadding(padW, padW, padH, padH))
  if (kdim == 2) {
    meanestimator.add(new SpatialConvolution(nInputPlane, 1, kernel.size(2), kernel.size(1)))
  } else {
    meanestimator.add(new SpatialConvolutionMap(SpatialConvolutionMap.oneToOne(nInputPlane), kernel.size(1), 1))
    meanestimator.add(new SpatialConvolution(nInputPlane, 1, 1, kernel.size(1)))
  }
  meanestimator.add(new Replicate(nInputPlane,1,3))

  // set kernel(parameters._1(0)) and bias(parameters._1(1))
  if (kdim == 2) {
    for (i <- 1 to nInputPlane) {
      meanestimator.modules(1).parameters()._1(0)(1)(1)(i).copy(kernel)
    }
    meanestimator.modules(1).parameters()._1(1).zero()
  } else {
    for (i <- 1 to nInputPlane) {
      meanestimator.modules(1).parameters()._1(0)(i).copy(kernel)
      meanestimator.modules(2).parameters()._1(0)(1)(1)(i).copy(kernel)
    }
    meanestimator.modules(1).parameters()._1(1).zero()
    meanestimator.modules(2).parameters()._1(1).zero()
  }

  // other operation
  val subtractor = new CSubTable()
  val divider = new CDivTable()

  // coefficient array, to adjust side effects
  var coef = Tensor(1,1,1)

  @transient private var ones: Tensor[T] = null
  @transient private var adjustedsums: Tensor[T] = null
  @transient private var localsums: Tensor[T] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim = input.dim()
    if (input.dim()+1 != coef.dim() || (input.size(dim) != coef.size(dim)) || (input.size(dim-1) != coef.size(dim-1))) {
      if (null == ones) ones = Tensor[T]()
      if (dim == 4) {
        // batch mode
        ones.resizeAs(input(1)).fill(ev.fromType[Int](1))
        val _coef = meanestimator.updateOutput(ones)
        val size = Array(input.size(1)) ++ _coef.size()
        coef = coef.resizeAs(_coef).copy(_coef).view(Array(1) ++ _coef.size()).expand(size)
      } else {
        ones.resizeAs(input).fill(ev.fromType[Int](1))
        val _coef = meanestimator.updateOutput(ones)
        coef.resizeAs(_coef).copy(_coef)
      }
    }

    // compute mean
    localsums = meanestimator.updateOutput(input)
    adjustedsums = divider.updateOutput(T(localsums, coef))
    output = subtractor.updateOutput(T(input, adjustedsums))

    output
  }


  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    // resize grad
    gradInput.resizeAs(input).zero()

    // backprop through all modules
    val gradsub = subtractor.updateGradInput(T(input, adjustedsums), gradOutput)
    val graddiv = divider.updateGradInput(T(localsums, coef), gradsub(2))
    val size = meanestimator.updateGradInput(input, graddiv(1)).size()
    gradInput.add(meanestimator.updateGradInput(input, graddiv(1)))
    gradInput.add(gradsub[Tensor[T]](1))

    gradInput
  }

  override def toString(): String = {
    s"SpatialSubtractiveNormalization($nInputPlane, kernelTensor)"
  }

}
