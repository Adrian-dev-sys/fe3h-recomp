// Mods tab (Settings, start screen and in game): the folders in mods/ (runtime/mods.c) - on/off, load order by drag
// and drop (or the arrows), and a warning when enabled mods change the same files.
#include <SDL3/SDL.h>
#include <string>
#include <sys/stat.h>

#include "imgui.h"
#include "theme.h"
#include "mods.h"
#include "paths.h"
#include "platform.h"

static const ImVec4 kWarn(1, 0.72f, 0.4f, 1);

static void open_mods_folder(void) {
    std::string path = std::string(rt_data_root()) + "/mods";
    mkdir(path.c_str(), 0755);
    std::string url = "file://";
    if (path[0] != '/') url += "/";  // (Windows: file:///C:/...)
    for (char ch : path) {
        if (ch == '\\') ch = '/';
        if (ch == ' ') url += "%20";
        else if (ch == '%') url += "%25";
        else if (ch == '#') url += "%23";
        else url += ch;
    }
    SDL_OpenURL(url.c_str());
}

// "X and Y both change N files (e.g. path); Y wins (lower in the list)"
static std::string conflict_text(int a, int b) {
    const char *ex = "";
    int n = mods_conflict(a, b, &ex);
    if (!n) return "";
    ModInfo x, y;
    mods_info(a, &x), mods_info(b, &y);
    char buf[1200];
    snprintf(buf, sizeof buf, "%s and %s both change %d file%s (e.g. %s). %s wins: it's lower in the list.", x.name,
             y.name, n, n == 1 ? "" : "s", ex, (a > b ? x : y).name);
    return buf;
}

extern "C" void mods_tab_draw(float scale, bool in_game) {
    int n = mods_count();
    static std::string popup_msg;
    bool open_popup = false;
    // enabled pairs that change the same files
    int npairs = 0;
    for (int a = 0; a < n; a++)
        for (int b = a + 1; b < n; b++) {
            ModInfo x, y;
            if (mods_info(a, &x) && mods_info(b, &y) && x.on && y.on && mods_conflict(a, b, NULL)) npairs++;
        }
    ImGui::PushTextWrapPos(0);
    ImGui::TextDisabled("Tick the mods to use and drag them into order: lower in the list wins when two change the same "
                        "file. Changes apply the next time the game starts%s.",
                        in_game ? " (Quit to Start Screen, then Play)" : "");
    if (npairs)
        ImGui::TextColored(kWarn, "Warning: %d pair%s of enabled mods change the same files - see below. Mods that "
                           "change the same game data usually don't work together.", npairs, npairs == 1 ? "" : "s");
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    if (n == 0) ImGui::TextUnformatted("No mods found.");
    float arrows = ImGui::GetFrameHeight() * 2 + ImGui::GetStyle().ItemSpacing.x * 2;
    int move_from = -1, move_to = -1;
    for (int i = 0; i < n; i++) {
        ModInfo mi;
        if (!mods_info(i, &mi)) continue;
        ImGui::PushID(i);
        bool on = mi.on;
        if (ImGui::Checkbox("##on", &on)) {
            mods_set(i, on);
            if (on) {  // turning one on that clashes with an enabled mod: say so now
                popup_msg.clear();
                for (int k = 0; k < n; k++) {
                    ModInfo o;
                    if (k == i || !mods_info(k, &o) || !o.on) continue;
                    std::string t = conflict_text(i < k ? i : k, i < k ? k : i);
                    if (!t.empty()) popup_msg += t + "\n\n";
                }
                open_popup = !popup_msg.empty();
            }
        }
        ImGui::SameLine();
        char label[200];
        snprintf(label, sizeof label, "%d.  %s", i + 1, mi.name);
        ImGui::PushStyleColor(ImGuiCol_Text, mi.on ? THEME_TEXT : THEME_QUIET);
        ImGui::Selectable(label, false, 0, ImVec2(ImGui::GetContentRegionAvail().x - arrows, 0));
        ImGui::PopStyleColor();
        {  // what it contains: small, at the row's right end (romfs: game data, dlc, exefs: code)
            std::string parts;
            if (mi.romfs) parts += "game data";
            if (mi.dlc) parts += parts.empty() ? "DLC" : ", DLC";
            if (mi.exefs) parts += parts.empty() ? "code" : ", code";
            if (parts.empty()) parts = "empty";
            float fs = ImGui::GetFontSize() * 0.72f;
            ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
            ImVec2 sz = ImGui::GetFont()->CalcTextSizeA(fs, FLT_MAX, 0, parts.c_str());
            ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), fs, ImVec2(b.x - sz.x - 8 * scale, (a.y + b.y - sz.y) * 0.5f),
                                                ImGui::GetColorU32(ImGuiCol_TextDisabled), parts.c_str());
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            if (!ImGui::IsMouseDown(0))
                ImGui::SetTooltip("Drag to reorder. game data: files in its romfs folder; DLC: DLC files; code: exefs patches.");
        }
        if (ImGui::BeginDragDropSource()) {
            ImGui::SetDragDropPayload("MOD_ROW", &i, sizeof i);
            ImGui::Text("Move %s", mi.name);
            ImGui::EndDragDropSource();
        }
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload *p = ImGui::AcceptDragDropPayload("MOD_ROW")) move_from = *(const int *)p->Data, move_to = i;
            ImGui::EndDragDropTarget();
        }
        ImGui::SameLine();  // (controller / keyboard: the arrows)
        ImGui::BeginDisabled(i == 0);
        if (ImGui::ArrowButton("##up", ImGuiDir_Up)) move_from = i, move_to = i - 1;
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(i == n - 1);
        if (ImGui::ArrowButton("##down", ImGuiDir_Down)) move_from = i, move_to = i + 1;
        ImGui::EndDisabled();
        ImGui::Indent(ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.x);

        if (mi.needs_gen)
            ImGui::TextColored(kWarn, "Its code patches aren't built in yet: Game Files > Rebuild game code.");
        if (mi.on) {
            ImGui::PushTextWrapPos(0);
            for (int k = 0; k < n; k++) {
                ModInfo o;
                if (k == i || !mods_info(k, &o) || !o.on) continue;
                const char *ex = "";
                int c = mods_conflict(i, k, &ex);
                if (c) ImGui::TextColored(kWarn, "Conflicts with %s: %d file%s (e.g. %s). %s wins.", o.name, c,
                                          c == 1 ? "" : "s", ex, k > i ? o.name : "This one");
            }
            ImGui::PopTextWrapPos();
        }
        ImGui::Unindent(ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.x);
        ImGui::Spacing();
        ImGui::PopID();
    }
    if (move_from >= 0) mods_move(move_from, move_to);
    ImGui::Spacing();
    if (ImGui::Button("Open mods folder")) open_mods_folder();
    ImGui::SameLine();
    if (ImGui::Button("Refresh")) mods_rescan();
    ImGui::PushTextWrapPos(0);
    ImGui::TextDisabled("One folder per mod, in the Switch mod layout (as for Ryujinx / yuzu): mods/<name>/romfs/... for "
                        "game files, mods/<name>/exefs/*.pchtxt or *.ips for code patches, "
                        "mods/<name>/dlc/<content id>/romfs/... for DLC files.");
    ImGui::PopTextWrapPos();
    if (open_popup) ImGui::OpenPopup("Mod conflict");
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 30, 0), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("Mod conflict", NULL, ImGuiWindowFlags_NoSavedSettings)) {
        ImGui::TextColored(kWarn, "These mods change the same files");
        ImGui::PushTextWrapPos(0);
        ImGui::TextUnformatted(popup_msg.c_str());
        ImGui::TextDisabled("Both stay on. Turn one off, or order them so the one you want wins.");
        ImGui::PopTextWrapPos();
        if (ImGui::Button("OK", ImVec2(ImGui::GetFontSize() * 8, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false) ||
            ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}
