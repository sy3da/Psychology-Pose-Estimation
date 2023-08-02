#version 330

layout(location = 0) in vec4 position_worldspace;
layout(location = 1) in vec2 vertex_texcoords;

out vec2 texcoords;

uniform mat4 MVP;


void main()
{
	gl_Position = MVP *  position_worldspace ;
	texcoords = vertex_texcoords;
}
