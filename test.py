import torch

pixel_locations = torch.meshgrid(torch.arange(0, 512), torch.arange(0, 512))
    # 在最里层新建维度上进行拼接，然后reshape再repeat
pixel_locations = torch.stack(
        [pixel_locations[0], pixel_locations[1]],
        dim=-1).long().view(1, -1, 2).repeat(4, 1, 1)
pixel_locations[..., -1] *= -1
pixel_locations=pixel_locations.permute(0, 2, 1)
pixel_locations=torch.cat([pixel_locations, torch.ones_like(pixel_locations)], dim=1)
print(pixel_locations)
