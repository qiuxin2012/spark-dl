/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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
package com.intel.analytics.bigdl.dataset

import org.apache.commons.lang3.SerializationUtils

import scala.collection.Iterator

/**
 * Transform a data stream of type A to type B. It is usually used in data pre-process stage.
 * Different transformers can compose a pipeline. For example, if there're transformer1 from A to
 * B, transformer2 from B to C, and transformer3 from C to D, you can compose them into a bigger
 * transformer from A to D by transformer1 -> transformer2 -> transformer 3.
 *
 * The purpose of transformer is for code reuse. Many deep learning share many common data
 * pre-process steps. User needn't write them every time, but can reuse others work.
 *
 * Transformer can be used with RDD(rdd.mapPartition), iterator and DataSet.
 * @tparam A
 * @tparam B
 */
trait Transformer[A, B] extends Serializable {
  def apply(prev: Iterator[A]): Iterator[B]

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> [C](other: Transformer[B, C]): Transformer[A, C] = {
    new ChainedTransformer(this, other)
  }

  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName

  def cloneTransformer(): Transformer[A, B] = {
    SerializationUtils.clone(this)
  }
}

class ChainedTransformer[A, B, C](first: Transformer[A, B], last: Transformer[B, C])
  extends Transformer[A, C] {
  override def apply(prev: Iterator[A]): Iterator[C] = {
    last(first(prev))
  }
}

object Identity {
  def apply[A](): Identity[A] = new Identity[A]()
}

class Identity[A] extends Transformer[A, A] {
  override def apply(prev: Iterator[A]): Iterator[A] = {
    prev
  }
}
