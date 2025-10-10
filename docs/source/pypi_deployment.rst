PyPI Deployment
===============

This project uses automated deployment to PyPI using GitHub Actions.

Publishing
----------

To publish a new version:

1. **Create a version tag:**
   .. code-block:: bash

      git tag v1.0.0
      git push origin v1.0.0

2. **Automatic publishing:**
   The GitHub Actions workflow will automatically build and publish to PyPI.
