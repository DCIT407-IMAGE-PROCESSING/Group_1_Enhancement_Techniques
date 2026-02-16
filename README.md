# Group_1_Enhancement_Techniques
Focus on grayscale vs color, resolution, quantization and simple enhancement



## IMAGE REPRESENTATION

## 1. Introduction

An image, in everyday terms, is a visual representation of a scene. It may be a photograph, a scanned document, or any visual content captured by a camera or sensor. However, a computer does not interpret an image as a “picture.” Instead, it represents and processes an image as structured numerical data.

A **digital image** is a discrete numerical representation of a visual scene. It consists of a finite set of values that describe the intensity of light at specific spatial locations. Understanding this numerical representation is fundamental to image processing, since all image operations are performed on these values.




## 2. Pixels and the Image Grid

### 2.1 Pixel (Picture Element)

A **pixel** (short for picture element) is the smallest addressable unit of a digital image. Each pixel stores numerical information that represents the intensity or color at a specific location in the image.

In grayscale images, a pixel stores a single intensity value. In color images, a pixel typically stores multiple values corresponding to different color channels.

### 2.2 The Image Grid

A digital image is organized as a two-dimensional grid of pixels arranged in rows and columns. Each pixel occupies a specific position in this grid and is identified by spatial coordinates.

If we denote:

* ( x ) as the horizontal coordinate (column index), and
* ( y ) as the vertical coordinate (row index),

then each pixel location can be represented as (x, y).

In most digital image systems:

* The origin (0,0) is located at the **top-left corner**.
* The ( x )-coordinate increases from left to right.
* The ( y )-coordinate increases from top to bottom.

Thus, an image is not simply a collection of numbers; it is a **spatially structured array of numbers**, where each value corresponds to a specific location in the scene.

---



## 3. Mathematical Representation of a Digital Image

To model an image formally, we represent it as a two-dimensional discrete function:
f(x, y)

where:
* (x, y) denotes a spatial coordinate in the image grid,
* f(x, y) gives the intensity value at that coordinate.

For a grayscale digital image, this can be written more formally as:

f : Z² → {0, 1, 2, ..., L−1}

where:
* Z² represents the set of discrete integer coordinate pairs,
* L is the number of possible intensity levels.

The **domain** of the function is the set of all pixel coordinates in the image.
The **range** is the set of possible intensity values.

For example, in an 8-bit grayscale image:
L = 256
and pixel intensity values range from 0 (black) to 255 (white).

This mathematical representation is fundamental because it allows image processing operations to be expressed as transformations of the function f(x, y).

---

## 4. Matrix Representation of a Digital Image
While the function f(x, y) provides a conceptual model, a digital image is implemented in practice as a matrix (or two-dimensional array).

An image of size ( 3 × 3 ) can be represented as:

I = [  
  [ f(0,0)  f(1,0)  f(2,0) ]  
  [ f(0,1)  f(1,1)  f(2,1) ]  
  [ f(0,2)  f(1,2)  f(2,2) ]  
]

In this representation:

* Each **row** corresponds to a fixed ( y )-value.
* Each **column** corresponds to a fixed ( x )-value.
* Each matrix entry stores the intensity value of a pixel.
