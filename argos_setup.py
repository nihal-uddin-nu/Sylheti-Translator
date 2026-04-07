import argostranslate.package
import argostranslate.translate

# Update package index and install model (run once)
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    p for p in available_packages
    if p.from_code == "en" and p.to_code == "bn"
)
argostranslate.package.install_from_path(package_to_install.download())

# Perform translation
translated = argostranslate.translate.translate("Hello World", "en", "bn")
print(translated)   