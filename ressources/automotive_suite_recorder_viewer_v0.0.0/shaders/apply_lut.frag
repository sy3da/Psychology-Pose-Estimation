#version 330 core

in vec2 texcoords;
out vec4 color;

uniform sampler2D input_image;
uniform sampler2D colormap;
uniform float min_val;
uniform float max_val;

void main()
{
	vec2 uv = texcoords.xy;
  
	vec4 bkg_color = texture(input_image, uv);

	float value = bkg_color.x;
	value = clamp(value, min_val, max_val);

	float range = max_val - min_val;
	value = (value - min_val) / range;

		
	value = clamp(value, 0.0001, 0.9999);
	color = texture(colormap, vec2(value, 0));

}

