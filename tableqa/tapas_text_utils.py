# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
# coding=utf8
"""Includes utility functions to convert and normalize text."""

import math
import re
import unicodedata
import six


def wtq_normalize(x):
  """Returns the normalized version of x.

  This normalization function is taken from WikiTableQuestions github, hence the
  wtq prefix. For more information, see
  https://github.com/ppasupat/WikiTableQuestions/blob/master/evaluator.py

  Args:
    x: the object (integer type or string) to normalize.

  Returns:
    A normalized string.
  """
  x = x if isinstance(x, six.text_type) else six.text_type(x)
  # Remove diacritics.
  x = "".join(
      c for c in unicodedata.normalize("NFKD", x)
      if unicodedata.category(c) != "Mn")
  # Normalize quotes and dashes.
  x = re.sub(u"[‘’´`]", "'", x)
  x = re.sub(u"[“”]", '"', x)
  x = re.sub(u"[‐‑‒−]", "-", x)
  x = re.sub(u"[‐]", "", x)
  while True:
    old_x = x
    # Remove citations.
    x = re.sub(u"((?<!^)\\[[^\\]]*\\]|\\[\\d+\\]|[♦†‡*#+])*$", "",
               x.strip())
    # Remove details in parenthesis.
    x = re.sub(u"(?<!^)( \\([^)]*\\))*$", "", x.strip())
    # Remove outermost quotation mark.
    x = re.sub(u'^"([^"]*)"$', r"\1", x.strip())
    if x == old_x:
      break
  # Remove final '.'.
  if x and x[-1] == ".":
    x = x[:-1]
  # Collapse whitespaces and convert to lower case.
  x = re.sub(r"\s+", " ", x, flags=re.U).lower().strip()
  x = re.sub("<[^<]+?>", "", x)
  x = x.replace("\n", " ")
  return x


_TOKENIZER = re.compile(r"\w+|[^\w\s]+", re.UNICODE)


def tokenize_string(x):
  return list(_TOKENIZER.findall(x.lower()))


# List of string normalization functions to be applied in order. We go from
# simplest to more complex normalization procedures.
STRING_NORMALIZATIONS = (
    lambda x: x,
    lambda x: x.lower(),
    tokenize_string,
    wtq_normalize,
)


def _split_thousands(delimiter, value):
  split = value.split(delimiter)
  return len(split) > 1 and any(map(lambda x: len(x) == 3, split))


def convert_to_float(value):
  """Converts value to a float using a series of increasingly complex heuristics.

  Args:
    value: object that needs to be converted. Allowed types include
      float/int/strings.

  Returns:
    A float interpretation of value.

  Raises:
    ValueError if the float conversion of value fails.
  """
  if isinstance(value, float):
    return value
  if isinstance(value, int):
    return float(value)
  if not isinstance(value, six.string_types):
    raise ValueError("Argument value is not a string. Can't parse it as float")
  sanitized = value

  try:
    # Example: 1,000.7
    if "." in sanitized and "," in sanitized:
      return float(sanitized.replace(",", ""))
    # 1,000
    if "," in sanitized and _split_thousands(",", sanitized):
      return float(sanitized.replace(",", ""))
    # 5,5556
    if "," in sanitized and sanitized.count(",") == 1 and not _split_thousands(
        ",", sanitized):
      return float(sanitized.replace(",", "."))
    # 0.0.0.1
    if sanitized.count(".") > 1:
      return float(sanitized.replace(".", ""))
    # 0,0,0,1
    if sanitized.count(",") > 1:
      return float(sanitized.replace(",", ""))
    return float(sanitized)
  except ValueError:
    # Avoid adding the sanitized value in the error message.
    raise ValueError("Unable to convert value to float")


def _normalize_float(answer):
  if answer is None:
    return None
  try:
    value = convert_to_float(answer)
    if isinstance(value, float) and math.isnan(value):
      return None
    return value
  except ValueError:
    return answer.lower()


def normalize_answers(answers):
  normalized_answers = (_normalize_float(a) for a in answers)
  normalized_answers = (a for a in normalized_answers if a is not None)
  normalized_answers = (str(a) for a in normalized_answers)
  normalized_answers = list(normalized_answers)
  normalized_answers.sort()
  return normalized_answers
