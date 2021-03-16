module CellSegmentation

import AxisArrays
import ColorTypes
import ImageBinarization
import ImageFiltering
import ImageMorphology
import ImageSegmentation
import Interpolations

using Statistics: quantile, median 

export
    CellImageAxes,
    CellImage,
    CellImageNoAxes,
    cell_image,
    median_filter,
    get_background_mask,
    remove_edges,
    nearest_neighbor_resize,
    segment_via_background,
    extract_segment

# Type for an axis to index a cell image: a Height x Width x Channel axes,
#   where the Height and Width dimensions are integer-indexed and Channels are
#   named by symbols 
CellImageAxes = Tuple{
    AxisArrays.Axis{:height,UnitRange{Int64}},
    AxisArrays.Axis{:width,UnitRange{Int64}},
    AxisArrays.Axis{:channel,Array{Symbol,1}}
}

# Type for a cell image: an AxisArray indexed using a CellImageAxis
CellImage = (
    AxisArrays.AxisArray{T,3,S,CellImageAxes}
    where S <: Array{T,3}
    where T <: ColorTypes.Gray{R}
    where R <: Union{ColorTypes.Fractional, Bool}
)

# Certain operations may strip axes, making this a useful type alias
CellImageNoAxes = (
    Array{T,3} where T <: ColorTypes.Gray{S} where S <: ColorTypes.Fractional
)


"""
Wrap an array containing a cell image into a nicely-indexed CellImage.
"""
function cell_image(
    image::CellImageNoAxes,
    channel_names::AbstractArray{Symbol,1}
)::CellImage
    (height, width, _) = size(image)
    return AxisArrays.AxisArray(
        image,
        AxisArrays.Axis{:height}(1:height),
        AxisArrays.Axis{:width}(1:width),
        AxisArrays.Axis{:channel}(channel_names)
    )
end


"""
Smooth an image using a windowed median, with window length as a fraction of image size.
"""
function median_filter(
    image::AbstractArray;
    window_size::Real=0.01,
    border_size::Real=0.1
)::AbstractArray
    # determine window size
    height, width = size(image)
    window_px = Int(floor(window_size * (height + width) / 2))
    if window_px % 2 == 0
        window_px += 1
    end
    pad_px = Int(round(border_size * (height + width) / 2))
    
    # apply filter
    border = ImageFiltering.Fill(0, pad_px)
    ImageFiltering.mapwindow(median, image, (window_px, window_px); border=border)
end


"""
Finds the background of a grayscale image, resulting in a black and white image.

Binarizes based on quantile of nonzero values.

Uses median smoothing and morphological opening for improved smoothness.

An alternative to something like
    `.!ImageBinarization.binarize(image, ImageBinarization.Otsu())``
"""
function get_background_mask(
    image::AbstractArray;
    median_filter_window_size::Real=0.005,
    background_nonzero_quantile::Real=0.2,
    area_opening_window_size::Real=0.02
)::AbstractArray
    # median smoothing smooth
    image = median_filter(image; window_size=median_filter_window_size)

    # threshold
    background_threshold = quantile(image[image .> 0], background_nonzero_quantile)
    is_background = image .< background_threshold

    # close some holes
    height, width = size(image)
    hole_area_px = Int(round(area_opening_window_size * (height + width) / 2))^2
    is_background = ImageMorphology.area_opening(is_background; min_area=hole_area_px)
end


"""
Finds, isolates, and removes edges in a grayscale image.

Potentially useful to increase distance between bordering cell nuclei.

TODO: Examine if Otsu's method is best binarization for edges.
TODO: Make edge thickening (by morphological dilation) independent of image resolution.
"""
function remove_edges(
    image::AbstractArray;
    min_edge_diameter_size::Real=0.04,
    n_dilations::Integer=2
)::AbstractArray
    # use edge detection to find edges
    kern_y, kern_x = ImageFiltering.Kernel.ando5()
    grad_y = ImageFiltering.imfilter(image, kern_y)
    grad_x = ImageFiltering.imfilter(image, kern_x)
    grad = (grad_y.^2) .+ (grad_x.^2)

    # binarize edges
    edge_mask = ImageBinarization.binarize(Bool, grad, ImageBinarization.Otsu())

    # drop tiny edge pieces
    height, width = size(image)
    min_diameter_px = Int(round(min_edge_diameter_size * (height + width) / 2))
    edge_mask = ImageMorphology.diameter_opening(
        edge_mask; min_diameter=min_diameter_px
    )

    # thicken edges
    edge_mask = ImageMorphology.dilate(edge_mask)

    res = copy(image)
    res[edge_mask] .= zero(eltype(res))
    return res
end


"""
Resize using nearest-neighbor Interpolation rather than
whatever ImageTransformations.imresize uses (which averages out integers).
"""
function nearest_neighbor_resize(
    image::AbstractArray,
    new_size::Tuple{Integer, Integer}
)::AbstractArray
    h_old, w_old = size(image)
    h_new, w_new = new_size
    h_range = LinRange(1, h_old, h_new)
    w_range = LinRange(1, w_old, w_new)
    int = Interpolations.interpolate(
        image, Interpolations.BSpline(Interpolations.Constant())
    )
    result = int(h_range, w_range)
end


"""
Segment nucleui or cells using marked watershed method.

If markers are not provided, they are established using a distance from the
    background, thresholded to a quantile of foreground distances from background.

Using a different background for markers is supported as well.
"""
function segment_via_background(
    background::AbstractArray;
    marker_background::Union{AbstractArray, Nothing}=nothing,
    markers::Union{AbstractArray, Nothing}=nothing,
    watershed_marker_quantile::Real=0.4
)::ImageSegmentation.SegmentedImage
    dist = (
        background
        |> ImageMorphology.feature_transform
        |> ImageMorphology.distance_transform
    )
    if !isa(markers, Nothing)
        if !isa(marker_background, Nothing)
            throw(
                ArgumentError("Cannot specify both `marker_background` and `markers`")
            )
        end
    else
        if isa(marker_background, Nothing)
            marker_dist = dist
        else
            marker_dist = (
                marker_background
                |> ImageMorphology.feature_transform
                |> ImageMorphology.distance_transform
            )
        end
        marker_threshold = quantile(
            marker_dist[marker_dist .> 0], watershed_marker_quantile
        )
        markers = ImageMorphology.label_components(marker_dist .>= marker_threshold)
    end
    segmentation = ImageSegmentation.watershed(-dist, markers, mask=.!background)
end


"""
Extract the contents of a segment mask and place it on a black background.
"""
function extract_segment(
    segment_mask::AbstractArray,
    full_image::CellImage
)::CellImage
    segment_indices = findall(segment_mask)
    idx_min = minimum(segment_indices)
    idx_max = maximum(segment_indices)
    shifted_indices = segment_indices .- (idx_min - CartesianIndex(1, 1),)
    max_dim = maximum((idx_max - idx_min).I) + 1
    channel_names = full_image.axes[3].val
    extracted_image = zeros(
        ColorTypes.Gray{ColorTypes.N0f8},
        (max_dim, max_dim, length(channel_names))
    )
    extracted_image = cell_image(extracted_image, channel_names)
    extracted_image[shifted_indices, :] = full_image[segment_indices, :]
    return extracted_image
end


end
