#include "CellGrowthScene.h"

class Application : public Elysium::Application
{
private:
    bool m_VSync = false;

public:
    Application(bool imgui=false) : Elysium::Application(imgui)
    {
        m_Window->setVSync(m_VSync);
        m_SceneManager.loadScene(new CellGrowthScene(m_Window->getWidth(), m_Window->getHeight()));
    }

    ~Application()
    {
    }

    void ApplicationLogic() override
    {
        ImGui::Begin("Main Application");
        ImGui::Checkbox("VSync", &m_VSync);
        ImGui::ColorEdit4("Clear Color", m_ClearColor);
        if (ImGui::Button("Generate New Grid"))
        {
            m_SceneManager.unloadScene();
            m_SceneManager.loadScene(new CellGrowthScene(m_Window->getWidth(), m_Window->getHeight()));
        }
        ImGui::End();

        m_Window->setVSync(m_VSync);
    }
};

int main(void)
{
    Application* application = new Application(true);
    application->Run();
    delete application;
    return 0;
}