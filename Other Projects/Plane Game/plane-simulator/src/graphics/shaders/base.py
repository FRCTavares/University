# FILE: /plane-simulator/plane-simulator/src/graphics/shaders/base.py
import OpenGL.GL.shaders

class Shader:
    def __init__(self, vertex_code, fragment_code):
        self.vertex_shader = self.compile_shader(vertex_code, OpenGL.GL.GL_VERTEX_SHADER)
        self.fragment_shader = self.compile_shader(fragment_code, OpenGL.GL.GL_FRAGMENT_SHADER)
        self.shader_program = self.create_program(self.vertex_shader, self.fragment_shader)

    def compile_shader(self, code, shader_type):
        shader = OpenGL.GL.glCreateShader(shader_type)
        OpenGL.GL.glShaderSource(shader, code)
        OpenGL.GL.glCompileShader(shader)
        if OpenGL.GL.glGetShaderiv(shader, OpenGL.GL.GL_COMPILE_STATUS) != OpenGL.GL.GL_TRUE:
            raise RuntimeError(OpenGL.GL.glGetShaderInfoLog(shader).decode())
        return shader

    def create_program(self, vertex_shader, fragment_shader):
        program = OpenGL.GL.glCreateProgram()
        OpenGL.GL.glAttachShader(program, vertex_shader)
        OpenGL.GL.glAttachShader(program, fragment_shader)
        OpenGL.GL.glLinkProgram(program)
        OpenGL.GL.glDetachShader(program, vertex_shader)
        OpenGL.GL.glDetachShader(program, fragment_shader)
        return program

    def use(self):
        OpenGL.GL.glUseProgram(self.shader_program)

    def cleanup(self):
        OpenGL.GL.glDeleteShader(self.vertex_shader)
        OpenGL.GL.glDeleteShader(self.fragment_shader)
        OpenGL.GL.glDeleteProgram(self.shader_program)