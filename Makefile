.PHONY: help docs docs-clean docs-api docs-build docs-serve docs-watch

help:
	@echo "Documentation commands:"
	@echo "  make docs          - Build HTML documentation"
	@echo "  make docs-clean    - Clean build directory"
	@echo "  make docs-api      - Generate API documentation only"
	@echo "  make docs-build    - Build documentation (HTML)"
	@echo "  make docs-serve    - Serve documentation locally (requires sphinx-autobuild)"
	@echo ""
	@echo "Installation:"
	@echo "  pip install -e .[docs]  - Install documentation dependencies"

docs: docs-clean docs-api docs-build

docs-clean:
	@echo "Cleaning documentation build directory..."
	rm -rf docs/_build
	rm -rf docs/api/*.rst
	@find docs/api -name "*.rst" -not -name "index.rst" -delete 2>/dev/null || true

docs-api:
	@echo "Generating API documentation..."
	.venv/bin/python -m sphinx.ext.apidoc \
		-f \
		-e \
		--separate \
		-o docs/api \
		src/psr_magpy
	@echo "API documentation generated in docs/api/"

docs-build:
	@echo "Building HTML documentation..."
	.venv/bin/python -m sphinx \
		-b html \
		-d docs/_build/doctrees \
		docs \
		docs/_build/html \
		-W \
		--keep-going
	@echo ""
	@echo "Documentation built successfully!"
	@echo "Open docs/_build/html/index.html in your browser"

docs-serve:
	@echo "Serving documentation at http://localhost:8000"
	@.venv/bin/sphinx-autobuild \
		--host 0.0.0.0 \
		--port 8000 \
		--open-browser \
		docs \
		docs/_build/html || \
		echo "Install sphinx-autobuild: pip install sphinx-autobuild"

docs-watch: docs-serve

