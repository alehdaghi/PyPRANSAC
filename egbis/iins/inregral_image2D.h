//
// Created by mahdi on 29.08.19.
//

#ifndef INTEGRALNORMALEST_INREGRAL_IMAGE2D_H
#define INTEGRALNORMALEST_INREGRAL_IMAGE2D_H


#include <vector>
#include <eigen3/Eigen/Dense>

using DataType = float;
using IntegralType = double;

template <unsigned Dimension>
class IntegralImage2D
{
public:
    static const unsigned second_order_size = (Dimension * (Dimension + 1)) >> 1;
    using ElementType = Eigen::Matrix<IntegralType, Dimension, 1>;
    using SecondOrderType = Eigen::Matrix<IntegralType, second_order_size, 1>;

    /** \brief Constructor for an Integral Image
      * \param[in] compute_second_order_integral_images set to true if we want to compute a second order image
      */
    IntegralImage2D (bool compute_second_order_integral_images) :
            first_order_integral_image_ (),
            second_order_integral_image_ (),
            width_ (1),
            height_ (1),
            compute_second_order_integral_images_ (compute_second_order_integral_images)
    {
    }

    /** \brief Destructor */
    virtual
    ~IntegralImage2D () { }

    /** \brief sets the computation for second order integral images on or off.
      * \param compute_second_order_integral_images
      */
    void
    setSecondOrderComputation (bool compute_second_order_integral_images);

    /** \brief Set the input data to compute the integral image for
      * \param[in] data the input data
      * \param[in] width the width of the data
      * \param[in] height the height of the data
      * \param[in] element_stride the element stride of the data
      * \param[in] row_stride the row stride of the data
      */
    void
    setInput (const DataType * data,
              unsigned width, unsigned height, unsigned element_stride, unsigned row_stride);

    /** \brief Compute the first order sum within a given rectangle
      * \param[in] start_x x position of rectangle
      * \param[in] start_y y position of rectangle
      * \param[in] width width of rectangle
      * \param[in] height height of rectangle
      */
    ElementType
    getFirstOrderSum (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;

    /** \brief Compute the first order sum within a given rectangle
      * \param[in] start_x x position of the start of the rectangle
      * \param[in] start_y x position of the start of the rectangle
      * \param[in] end_x x position of the end of the rectangle
      * \param[in] end_y x position of the end of the rectangle
      */
    ElementType
    getFirstOrderSumSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;

    /** \brief Compute the second order sum within a given rectangle
      * \param[in] start_x x position of rectangle
      * \param[in] start_y y position of rectangle
      * \param[in] width width of rectangle
      * \param[in] height height of rectangle
      */
    SecondOrderType
    getSecondOrderSum (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;

    /** \brief Compute the second order sum within a given rectangle
      * \param[in] start_x x position of the start of the rectangle
      * \param[in] start_y x position of the start of the rectangle
      * \param[in] end_x x position of the end of the rectangle
      * \param[in] end_y x position of the end of the rectangle
      */
    SecondOrderType
    getSecondOrderSumSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;

    /** \brief Compute the number of finite elements within a given rectangle
      * \param[in] start_x x position of rectangle
      * \param[in] start_y y position of rectangle
      * \param[in] width width of rectangle
      * \param[in] height height of rectangle
      */
    unsigned
    getFiniteElementsCount (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;

    /** \brief Compute the number of finite elements within a given rectangle
      * \param[in] start_x x position of the start of the rectangle
      * \param[in] start_y x position of the start of the rectangle
      * \param[in] end_x x position of the end of the rectangle
      * \param[in] end_y x position of the end of the rectangle
      */
    unsigned
    getFiniteElementsCountSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;

private:
    using InputType = Eigen::Matrix<DataType, Dimension, 1>;

    /** \brief Compute the actual integral image data
      * \param[in] data the input data
      * \param[in] element_stride the element stride of the data
      * \param[in] row_stride the row stride of the data
      */
    void
    computeIntegralImages (const DataType * data, unsigned row_stride, unsigned element_stride);

    std::vector<ElementType, Eigen::aligned_allocator<ElementType> > first_order_integral_image_;
    std::vector<SecondOrderType, Eigen::aligned_allocator<SecondOrderType> > second_order_integral_image_;
    std::vector<unsigned> finite_values_integral_image_;

    /** \brief The width of the 2d input data array */
    unsigned width_;
    /** \brief The height of the 2d input data array */
    unsigned height_;

    /** \brief Indicates whether second order integral images are available **/
    bool compute_second_order_integral_images_;
};

/**
  * \brief partial template specialization for integral images with just one channel.
  */
template<>
class IntegralImage2D <1>
{
public:
    static const unsigned second_order_size = 1;
    using ElementType = IntegralType;
    using SecondOrderType = IntegralType;

    /** \brief Constructor for an Integral Image
      * \param[in] compute_second_order_integral_images set to true if we want to compute a second order image
      */
    IntegralImage2D (bool compute_second_order_integral_images) :
            first_order_integral_image_ (),
            second_order_integral_image_ (),

            width_ (1), height_ (1),
            compute_second_order_integral_images_ (compute_second_order_integral_images)
    {
    }

    /** \brief Destructor */
    virtual
    ~IntegralImage2D () { }

    /** \brief Set the input data to compute the integral image for
      * \param[in] data the input data
      * \param[in] width the width of the data
      * \param[in] height the height of the data
      * \param[in] element_stride the element stride of the data
      * \param[in] row_stride the row stride of the data
      */
    void
    setInput (const DataType * data,
              unsigned width, unsigned height, unsigned element_stride, unsigned row_stride);

    /** \brief Compute the first order sum within a given rectangle
      * \param[in] start_x x position of rectangle
      * \param[in] start_y y position of rectangle
      * \param[in] width width of rectangle
      * \param[in] height height of rectangle
      */
    ElementType
    getFirstOrderSum (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;

    /** \brief Compute the first order sum within a given rectangle
      * \param[in] start_x x position of the start of the rectangle
      * \param[in] start_y x position of the start of the rectangle
      * \param[in] end_x x position of the end of the rectangle
      * \param[in] end_y x position of the end of the rectangle
      */
    ElementType
    getFirstOrderSumSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;

    /** \brief Compute the second order sum within a given rectangle
      * \param[in] start_x x position of rectangle
      * \param[in] start_y y position of rectangle
      * \param[in] width width of rectangle
      * \param[in] height height of rectangle
      */
    SecondOrderType
    getSecondOrderSum (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;

    /** \brief Compute the second order sum within a given rectangle
      * \param[in] start_x x position of the start of the rectangle
      * \param[in] start_y x position of the start of the rectangle
      * \param[in] end_x x position of the end of the rectangle
      * \param[in] end_y x position of the end of the rectangle
      */
    SecondOrderType
    getSecondOrderSumSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;

    /** \brief Compute the number of finite elements within a given rectangle
      * \param[in] start_x x position of rectangle
      * \param[in] start_y y position of rectangle
      * \param[in] width width of rectangle
      * \param[in] height height of rectangle
      */
    unsigned
    getFiniteElementsCount (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;

    /** \brief Compute the number of finite elements within a given rectangle
      * \param[in] start_x x position of the start of the rectangle
      * \param[in] start_y x position of the start of the rectangle
      * \param[in] end_x x position of the end of the rectangle
      * \param[in] end_y x position of the end of the rectangle
      */
    unsigned
    getFiniteElementsCountSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;

private:
    //  using InputType = typename IntegralImageTypeTraits<DataType>::Type;

    /** \brief Compute the actual integral image data
      * \param[in] data the input data
      * \param[in] element_stride the element stride of the data
      * \param[in] row_stride the row stride of the data
      */
    void
    computeIntegralImages (const DataType * data, unsigned row_stride, unsigned element_stride);

    std::vector<ElementType, Eigen::aligned_allocator<ElementType> > first_order_integral_image_;
    std::vector<SecondOrderType, Eigen::aligned_allocator<SecondOrderType> > second_order_integral_image_;
    std::vector<unsigned> finite_values_integral_image_;

    /** \brief The width of the 2d input data array */
    unsigned width_;
    /** \brief The height of the 2d input data array */
    unsigned height_;

    /** \brief Indicates whether second order integral images are available **/
    bool compute_second_order_integral_images_;
};

template class IntegralImage2D<1>;
template class IntegralImage2D<3>;

#endif //INTEGRALNORMALEST_INREGRAL_IMAGE2D_H
