# ==========================
# INPUT
# ==========================
st.subheader("🔧 Spesifikasi HP")

# Preset spesifikasi cepat
st.markdown("**⚡ Pilih Preset Spesifikasi (Opsional):**")

preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

preset = None
with preset_col1:
    if st.button("📱 Entry Level", use_container_width=True):
        preset = "entry"
with preset_col2:
    if st.button("🚀 Mid Range", use_container_width=True):
        preset = "mid"
with preset_col3:
    if st.button("🔋 Battery Besar", use_container_width=True):
        preset = "battery"
with preset_col4:
    if st.button("🔄 Reset", use_container_width=True):
        preset = "reset"

# Set nilai default berdasarkan preset
if preset == "entry":
    default_ram = 3
    default_storage = 32
    default_battery = 2000
    default_camera = 8.0
    default_screen = 5.5
elif preset == "mid":
    default_ram = 6
    default_storage = 128
    default_battery = 4000
    default_camera = 48.0
    default_screen = 6.5
elif preset == "battery":
    default_ram = 4
    default_storage = 64
    default_battery = 6000
    default_camera = 13.0
    default_screen = 6.5
else:
    default_ram = 4
    default_storage = 128
    default_battery = 5000
    default_camera = 50.0
    default_screen = 6.5

st.divider()

col1, col2 = st.columns(2)

with col1:
    screen_size = st.slider(
        "📐 Screen Size (inch)",
        min_value=4.0,
        max_value=8.0,
        value=float(default_screen),
        step=0.1
    )
    rear_camera = st.slider(
        "📷 Rear Camera (MP)",
        min_value=1.0,
        max_value=200.0,
        value=float(default_camera),
        step=1.0
    )
    internal_memory = st.select_slider(
        "💾 Internal Memory (GB)",
        options=[16, 32, 64, 128, 256, 512, 1024],
        value=default_storage
    )

with col2:
    ram = st.select_slider(
        "🧠 RAM (GB)",
        options=[1, 2, 3, 4, 6, 8, 12, 16, 32],
        value=default_ram
    )
    battery = st.select_slider(
        "🔋 Battery (mAh)",
        options=[1000, 2000, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 10000],
        value=default_battery
    )
