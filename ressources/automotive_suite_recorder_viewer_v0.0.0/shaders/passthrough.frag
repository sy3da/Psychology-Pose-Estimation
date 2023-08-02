#version 330 core

in vec2 texcoords;
out vec4 color;

uniform sampler2D input_image;

void main()
{
	color = texture(input_image, texcoords);
}

