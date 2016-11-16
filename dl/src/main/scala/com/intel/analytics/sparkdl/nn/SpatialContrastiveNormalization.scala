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

import scala.reflect.ClassTag

class SpatialContrastiveNormalization[T: ClassTag](
  val nInputPlane: Int = 1,
  var kernel: Tensor[T] = null,
  val threshold: Double = 1e-4,
  val thresval: Double = 1e-4
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  if (kernel == null) kernel = Tensor.ones[T](9, 9)

  private val kdim = kernel.nDimension()
  require(kdim == 1 || kdim == 2, "averaging kernel must be 2D or 1D")
  require(kernel.size(1) % 2 != 0, "averaging kernel must have ODD dimensions")
  if (kdim == 2) {
    require(kernel.size(2) % 2 != 0, "averaging kernel must have ODD dimensions")
  }

  // instantiate sub+div normalization
  val normalizer = new Sequential[Tensor[T], Tensor[T], T]()
  normalizer.add(new SpatialSubtractiveNormalization(nInputPlane, kernel))
  normalizer.add(new SpatialDivisiveNormalization(nInputPlane, kernel, threshold, thresval))

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = normalizer.forward(input)
    output
  }


  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = normalizer.backward(input, gradOutput)
    gradInput
  }

  override def toString(): String = {
    s"SpatialContrastiveNormalization($nInputPlane, kernelTensor, $threshold, $thresval)"
  }

}
