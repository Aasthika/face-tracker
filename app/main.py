from app.config_loader import ConfigLoader

def main():
    print("🚀 Starting Intelligent Face Tracker...")

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.get_all()

    print("✅ Config Loaded Successfully:")
    for key, value in config.items():
        print(f"{key}: {value}")

    print("\n✅ System initialized successfully!")
    print("👉 Next: Video Stream Module")

if __name__ == "__main__":
    main()