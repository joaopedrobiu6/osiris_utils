```python
%load_ext autoreload
%autoreload 2
```


```python
import osiris_utils as ou
```

# Interface with Osiris Input Decks

For many tasks, it might be useful to read simulation parameters directly from the input deck, or algorithmically generate new Osiris input decks (e.g.to run large parameter scans). 

The class `InputDeckIO` allows you to easily perform theses tasks.
You need to provide it two inputs:
- `filename`: path to OSIRIS input deck
- `verbose`: if `True` will print auxiliary information when parsing the input deck


## Reading an input deck


```python
deck = ou.InputDeckIO('example_data/thermal.1d', verbose=True)
```

Not only does the object load the different module parameters, but also automatically determines other useful information for downstream tasks.

Examples include:
- `dim`: number of dimensions
- `n_species`: number of species
- `species`: dictionary with `Species` objects with relevant information (e.g. charge, rqm, etc.)


```python
print("Simulation Dimensions:", deck.dim)
print("Number of species:", deck.n_species)
print("Species:", deck.species)
```

`InputDeckIO` stores the state of the input deck in the parameter `self._sections`.

You should not access this value directly, instead you can use for example its getter function `self.sections` which returns a deepcopy.

The returned value corresponds to a list of pairs [section_name, dictionary] where the section_name matches the Osiris input deck section name and he dictionary contains the list of (key, values) for the parameters of the section.


```python
print(deck.sections[0])
print(deck.sections[1])
print('Number of sections:', len(deck.sections))
```

Another option is to query the object as if it is an iterator, where the key is the section name.

This will return a list of dictionaries, since multiple sections can have the same name (e.g. you might have multiple species).

Once again, a deepcopy is being returned so editing the returned values of the dictionaries will not change the original `InputDeckIO` object.


```python
print(deck['simulation'])
print(deck['node_conf'])
```

Finally you can also ask for a specific parameter directly with `get_param()`


```python
# this one returns a list since there can be multiple sections with the same name
print(deck.get_param(section='simulation', param='random_seed')[0])
# which is equivalent to doing this
print(deck["simulation"][0]['random_seed'])
```

## Editing an input deck

To safely edit the value of a parameter in an input deck you can use `set_parameter()`.

**Note**: This re-writes the object values!



```python
# edit a parameter already exists
print('Before', deck['simulation'])
deck.set_param('simulation', 'random_seed', value=42)
print('After', deck['simulation'])
```


```python
# add a parameter
print("Before", deck["simulation"])
deck.set_param("simulation", "new_parameter", value='HI!', unexistent_ok=True)
print("After", deck["simulation"])
```

And you can also delete the parameter using `delete_param()`.


```python
# delete a parameter
print("Before", deck["simulation"])
deck.delete_param("simulation", "new_parameter")
print("After", deck["simulation"])
```

Something slighlty more powerful, is the ability to edit a string in the whole input deck (use with care!)

This can be done for example to automatically change multiple parameter values that are shared / depend on an external quantity.

We can do this using the function `set_tag()`.

**Note**: This only works on the parameter values, not section names / parameter names.


```python
# this a dummy example where we change some values to #tag#
print("Before", deck["simulation"])
deck.set_tag(
    '42', # this has to be a string
    '#tag#',
)
deck.set_tag(
    '23:50:00', # this has to be a string
    '#tag#',
)
print("After 1", deck["simulation"])

# and here we change both values at once
deck.set_tag('#tag#', "BOTH CHANGED")
print("After 2", deck["simulation"])
# There is a reason why wall_clock_limit has an extra "" done worry
# it is because it should be a string, while random_seed is an int!
# InputDeckIO handles these things for you
```

## Writing changes to file

Once you did your changes, you can simply generate a new input deck with `print_to_file()`.


```python
deck.print_to_file("edited-deck.1d")
! cat edited-deck.1d
```
