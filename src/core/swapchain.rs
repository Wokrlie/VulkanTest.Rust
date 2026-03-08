use anyhow::Result;
use winit::window::Window;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSwapchainExtensionDeviceCommands;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;

use crate::AppData;
use crate::QueueFamilyIndices;
use crate::core;


#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    pub unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice
    ) -> Result<Self> {
        Ok( Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(
                    physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(
                    physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(
                        physical_device, data.surface)?,
        })
    }
}

pub unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    old_swapchain: vk::SwapchainKHR,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .old_swapchain(old_swapchain);

    data.swapchain = device.create_swapchain_khr(&info, None)?;
    data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;
    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    Ok(())
}

pub unsafe fn create_swapchain_image_views(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|i| {
            let components = vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY);

            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let info = vk::ImageViewCreateInfo::builder()
                .image(*i)
                .view_type(vk::ImageViewType::_2D)
                .format(data.swapchain_format)
                .components(components)
                .subresource_range(subresource_range);

            device.create_image_view(&info, None)
        })
    .collect::<Result<Vec<_>, _>>()?;
    Ok(())
}

pub unsafe fn recreate_swapchain(instance: &Instance, device: &Device, window: &Window, data: &mut AppData) -> Result<()> {
    device.device_wait_idle()?;
    let old_swapchain = data.swapchain;
    cleanup_swapchain_resources(device, data);
    core::swapchain::create_swapchain(window, &instance, &device, data, old_swapchain)?;
    if old_swapchain != vk::SwapchainKHR::null() {
        device.destroy_swapchain_khr(old_swapchain, None);
    }
    core::swapchain::create_swapchain_image_views(&device, data)?;

    core::pipeline::create_render_pass(&instance, &device, data)?;
    core::pipeline::create_pipeline(&device, data)?;
    core::pipeline::create_framebuffers(&device, data)?;
    core::buffer::create_uniform_buffer(&instance, &device, data)?;
    core::commands::create_command_buffers(&device, data)?;
    data.images_in_flight.resize(data.swapchain_images.len(), vk::Fence::null());
    Ok(())
}

pub unsafe fn destroy_swapchain(device: &Device, data: &mut AppData){
    cleanup_swapchain_resources(device, data);
    device.destroy_swapchain_khr(data.swapchain, None);
}

unsafe fn cleanup_swapchain_resources(device: &Device, data: &mut AppData) {
    data.uniform_buffers.iter().for_each(|b| device.destroy_buffer(*b, None));
    data.uniform_buffers_memory.iter().for_each(|m| device.free_memory(*m, None));
    device.free_command_buffers(data.command_pool, &data.command_buffers);
    data.framebuffers.iter().for_each(|f| device.destroy_framebuffer(*f, None));
    device.destroy_pipeline(data.pipeline, None);
    device.destroy_pipeline_layout(data.pipeline_layout, None);
    device.destroy_render_pass(data.render_pass, None);
    data.swapchain_image_views.iter().for_each(|v| device.destroy_image_view(*v, None));
}

fn get_swapchain_surface_format(
    formats: &[vk::SurfaceFormatKHR],
) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
    .unwrap_or_else(|| formats[0])
}

fn get_swapchain_present_mode(
    present_modes: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(
    window: &Window,
    capabilities: vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D::builder()
            .width(window.inner_size().width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
            ))
            .height(window.inner_size().height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
            ))
            .build()
    }
}
