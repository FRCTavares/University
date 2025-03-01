# FILE: /plane-simulator/plane-simulator/src/graphics/shaders/terrain.py

# This file contains shader programs specific to terrain rendering.

vertex_shader_code = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

out vec2 fragTexCoord;
out vec3 fragNormal;
out vec3 fragPosition;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    fragTexCoord = texCoord;
    fragNormal = mat3(transpose(inverse(model))) * normal;
    fragPosition = vec3(model * vec4(position, 1.0));
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

fragment_shader_code = """
#version 330 core
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec3 fragPosition;

out vec4 color;

uniform sampler2D terrainTexture;
uniform vec3 lightPosition;
uniform vec3 viewPosition;

void main()
{
    // Ambient lighting
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * vec3(texture(terrainTexture, fragTexCoord));

    // Diffuse lighting
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lightPosition - fragPosition);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(texture(terrainTexture, fragTexCoord));

    // Specular lighting
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPosition - fragPosition);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * vec3(1.0);

    color = vec4(ambient + diffuse + specular, 1.0);
}
"""