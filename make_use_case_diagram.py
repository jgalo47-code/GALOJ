"""Draw assets/use_case_diagram.png with matplotlib."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis("off")

ax.add_patch(Rectangle((3.6, 0.3), 6.2, 7.4, fill=False, lw=1.8))
ax.text(6.7, 7.4, "Predictive Student Academic Risk Monitoring System", ha="center",
        fontsize=12, fontweight="bold")

cases = {
    "UC1": (6.7, 6.5, "Upload & validate dataset"),
    "UC2": (6.7, 5.4, "View exploratory analysis"),
    "UC3": (6.7, 4.3, "Predict student risk\n(single / batch)"),
    "UC4": (6.7, 3.2, "View risk dashboard\n& review queue"),
    "UC5": (6.7, 2.1, "Record final decision"),
    "UC6": (6.7, 1.0, "Export reports / logs"),
}
for key, (x, y, label) in cases.items():
    ax.add_patch(Ellipse((x, y), 4.2, 0.9, fill=True, fc="#eaf3fb", ec="#2e86ab", lw=1.5))
    ax.text(x, y, f"{key}: {label}", ha="center", va="center", fontsize=9)


def actor(x, y, name):
    ax.add_patch(plt.Circle((x, y + 0.55), 0.2, fill=False, lw=1.5))
    ax.plot([x, x], [y + 0.35, y - 0.2], "k", lw=1.5)
    ax.plot([x - 0.35, x + 0.35], [y + 0.2, y + 0.2], "k", lw=1.5)
    ax.plot([x, x - 0.3], [y - 0.2, y - 0.7], "k", lw=1.5)
    ax.plot([x, x + 0.3], [y - 0.2, y - 0.7], "k", lw=1.5)
    ax.text(x, y - 1.0, name, ha="center", fontsize=9, fontweight="bold")


actors = {"admin": (1.5, 6.0, "System\nAdministrator"),
          "instr": (1.5, 3.2, "Instructor /\nAdviser"),
          "coord": (11.8, 3.6, "Program\nCoordinator")}
for x, y, n in actors.values():
    actor(x, y, n)


def link(actor_key, uc_key):
    ax_, ay_, _ = actors[actor_key]
    ux, uy, _ = cases[uc_key]
    left = ax_ < 6
    ax.plot([ax_ + (0.4 if left else -0.4), ux + (-2.1 if left else 2.1)],
            [ay_ + 0.2, uy], color="#555", lw=1)


for uc in ["UC1", "UC2"]:
    link("admin", uc)
for uc in ["UC2", "UC3", "UC4", "UC5"]:
    link("instr", uc)
for uc in ["UC2", "UC4", "UC5", "UC6"]:
    link("coord", uc)

ax.text(6.7, 0.05, "Students have no direct access; support reaches them only through a human reviewer.",
        ha="center", fontsize=8.5, style="italic")
plt.tight_layout()
plt.savefig("assets/use_case_diagram.png", dpi=150)
print("saved")
