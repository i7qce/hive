# Run this script to add necessary environment variables and source commands to rc files
# MUST BE RUN IN THE SAME DIRECTORY THAT THE SCRIPT IS LOCATED IN

PANDOC_CONTAINER_DIR="$(pwd)"

if [[ $SHELL == /bin/zsh ]] ; then
    RC_FILE="$HOME/.zshrc"
elif [[ $SHELL == /bin/bash ]]; then
    RC_FILE="$HOME/.bashrc"
else
    echo "Unknown shell" >&2
    exit 1
fi

echo "RC file is: $RC_FILE"


echo "" >> "$RC_FILE"
echo "# required definitions and sources for docker pandoc aliases" >> "$RC_FILE"
echo "export MONOSPACE_TEMPLATE_DIR=$PANDOC_CONTAINER_DIR/monospace_template" >> "$RC_FILE"
echo "source $PANDOC_CONTAINER_DIR/pandoc_aliases.sh" >> "$RC_FILE"

source $RC_FILE

echo "Added necessary modifications to $RC_FILE"