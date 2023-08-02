#version 330 core

in vec2 texcoords;
out vec4 color;

uniform sampler2D input_image;

const float lut[5] = float[5](255.0 * (1.0/5), 255.0 * (2.0/5), 255.0 * (3.0/5), 255 * (4.0/5), 255.0 * (5.0/5));
const vec3 red = vec3(0.863, 0.196, 0.184);
const vec3 orange = vec3(0.796, 0.294, 0.086);
const vec3 yellow = vec3(0.710, 0.537, 0.000);
const vec3 magenta = vec3(0.827, 0.212, 0.510);
const vec3 green = vec3(0.522, 0.600, 0.000);
const vec3 cyan = vec3(0.165, 0.631, 0.596);

const vec3 cmap[5] = vec3[5](red, cyan, magenta, orange, green);

vec3 map_color(float value)
{
	vec3 color = cmap[0];
	for(int i=0; i<5; ++i)
	{
		if(value*255.0 < lut[i])
		{
			color=cmap[i];
		}
        else
        {
            break;
        }
	}
	return color;
}

void main()
{

	float value = texture(input_image, texcoords).x;
	//color = vec4(map_color(value), 1.f);
    color = vec4(value, value, 0.f, 1.f);
}

