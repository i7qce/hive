dpandoc () {
        echo "Running pandoc through docker-- note that all files must be located in the directory this command is run from (pwd), as that directory will be mounted in the container."
        docker run --rm --network none --volume "`pwd`:/data" --user `id -u`:`id -g` pandoc "$@"
}

monospace () {
        input_file=$1
        output_file="${input_file%.*}.html"

        input_file_dir=$(pwd)

        # Copy template directory into input file directory
        cp -r "$MONOSPACE_TEMPLATE_DIR" "$input_file_dir"

        dpandoc -f markdown+autolink_bare_uris --number-sections --toc -s --css monospace_template/reset.css --css monospace_template/index.css -i "$input_file" -o "$output_file" --template=monospace_template/template.html -V date="$(date +%b-%d-%Y)"
}