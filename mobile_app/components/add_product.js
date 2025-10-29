import { StatusBar } from 'expo-status-bar';
import React, { Component }  from 'react';
import { Linking, StyleSheet, Text, View, Button, Picker, Image, TextInput, TouchableOpacity, FlatList, ScrollView} from 'react-native';
import { Octicons, FontAwesome, AntDesign, Ionicons, Feather, Entypo } from '@expo/vector-icons';
import axios from 'axios';
// import {Backend_Url} from './backend_url'
import * as ImagePicker from 'expo-image-picker';
import DropDownPicker from 'react-native-dropdown-picker';
// secure store import
import * as SecureStore from 'expo-secure-store';

class AddProducts extends Component {
    constructor(props) {
        super(props);
        this.state = {
            user_access_token: '',
            image: null,
            image_extension: null,
            name: '',
            description: '',
            price: '',
            target_skin_types: null,
            target_skin_disorders: null,
            skin_types: [
                {label: 'Oily skin', value: 'Oily skin'},
                {label: 'Dry skin', value: 'Dry skin'},
                {label: 'Sensitive skin', value: 'Sensitive skin'},
                {label: 'Combination skin', value: 'Combination skin'},
                {label: 'Normal skin', value: 'Normal skin'}
            ],
            skin_disorders: [
                {label: 'Acne', value: 'Acne'},
                {label: 'Actinic keratoses', value: 'Actinic keratoses'},
                {label: 'Adult-onset dermatomyositis', value: 'Adult-onset dermatomyositis'},
                {label: 'Ageing skin', value: 'Ageing skin'},
                {label: 'Allergic contact dermatitis', value: 'Allergic contact dermatitis'},
                {label: 'Angioedema', value: 'Angioedema'},
                {label: 'Angiofibroma', value: 'Angiofibroma'},
                {label: 'Atopic dermatitis', value: 'Atopic dermatitis'},
                {label: 'Blepharitis', value: 'Blepharitis'},
                {label: 'Blushing', value: 'Blushing'},
                {label: 'Cutaneous lupus erythematosus', value: 'Cutaneous lupus erythematosus'},
                {label: 'Demodicosis', value: 'Demodicosis'},
                {label: 'Dermatitis', value: 'Dermatitis'},
                {label: 'Discoid lupus erythematosus', value: 'Discoid lupus erythematosus'}, 
                {label: 'Dry skin', value: 'Dry skin'},
                {label: 'Eczema', value: 'Eczema'},
                {label: 'Erythema dyschromicum perstans', value: 'Erythema dyschromicum perstans'},
                {label: 'Flushing', value: 'Flushing'},
                {label: 'Granuloma faciale', value: 'Granuloma faciale'},
                {label: 'Hair follicle tumours', value: 'Hair follicle tumours'},
                {label: 'Jessner lymphocytic infiltrate', value: 'Jessner lymphocytic infiltrate'},
                {label: 'Keratosis pilaris atrophicans faciei', value: 'Keratosis pilaris atrophicans faciei'},
                {label: 'Lichen planus pigmentosus', value: 'Lichen planus pigmentosus'},
                {label: 'Melasma', value: 'Melasma'},
                {label: 'Neonatal lupus erythematosus', value: 'Neonatal lupus erythematosus'},
                {label: 'Perioral/periorificial dermatitis', value: 'Perioral/periorificial dermatitis'},
                {label: 'Photosensitivity', value: 'Photosensitivity'},
                {label: 'Pityriasis alba', value: 'Pityriasis alba'},
                {label: 'Poikiloderma of Civatte', value: 'Poikiloderma of Civatte'},
                {label: 'Pseudofolliculitis barbae', value: 'Pseudofolliculitis barbae'},
                {label: 'Psoriasis', value: 'Psoriasis'},
                {label: 'Pyoderma faciale', value: 'Pyoderma faciale'},
                {label: 'Rosacea', value: 'Rosacea'},
                {label: 'Sarcoidosis', value: 'Sarcoidosis'},
                {label: 'Sebaceous hyperplasia', value: 'Sebaceous hyperplasia'},
                {label: 'Seborrhoea', value: 'Seborrhoea'},
                {label: 'Seborrhoeic dermatitis', value: 'Seborrhoeic dermatitis'},
                {label: 'Sensitive skin', value: 'Sensitive skin'},
                {label: 'Shaving rash', value: 'Shaving rash'},
                {label: 'Solar keratoses', value: 'Solar keratoses'},
                {label: 'Solar/senile comedones', value: 'Solar/senile comedones'},
                {label: 'Steroid acne', value: 'Steroid acne'},
                {label: 'Steroid rosacea', value: 'Steroid rosacea'},
                {label: 'Systemic lupus erythematosus', value: 'Systemic lupus erythematosus'},
                {label: 'Trigeminal trophic syndrome', value: 'Trigeminal trophic syndrome'},
                {label: 'Vitiligo', value: 'Vitiligo'}
            ],
            allergens: '',
            dropdown1_open: false,
            dropdown2_open: false
        }

        this.HandleChange = (value, state) => {
            this.setState({ [state]: value })
        }

        this.AddProduct = () => {
            var Backend_Url = this.state.backend_url
            var image = this.state.image
            var image_extension = this.state.image_extension
            var name = this.state.name
            var description = this.state.description
            var price = this.state.price
            var target_skin_types = this.state.target_skin_types
            var target_skin_disorders = this.state.target_skin_disorders
            var allergens = this.state.allergens

            if (image == null){
                alert('Image is required')
            }else if (name == ''){
                alert('Name is required')
            }else if (description == ''){
                alert('Description is required')
            }else if (price == ''){
                alert('Price is required')
            }else if (target_skin_types == ''){
                alert('Target skin types are required')
            }else if (target_skin_disorders == ''){
                alert('Target skin disorders are required')
            }else{

                var data = new FormData() 
                data.append('user_access_token', this.state.user_access_token)
                data.append('image', image)
                data.append('image_extension', image_extension)
                data.append('name', name)
                data.append('description', description)
                data.append('price', parseFloat(price))
                data.append('target_skin_types', target_skin_types)
                data.append('target_skin_disorders', target_skin_disorders)
                data.append('allergens', allergens)
                
                axios.post(Backend_Url + 'addProduct', data)
                .then((res) => {
                    let result = res.data
                    alert('Product added successfully.')
                    this.setState({
                        image: null,
                        image_extension: null,
                        name: '',
                        description: '',
                        price: '',
                        target_skin_types: null,
                        target_skin_disorders: null,
                        allergens: ''
                    })
                }).catch((error) => {
                    alert('Something went wrong, check your connection and try again.')
                })
            }

        }

        this.PickImage = async () => {
            console.log('***Picking product image***')

            // No permissions request is necessary for launching the image library
            let result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.All,
                allowsEditing: false,
                aspect: [4, 3],
                quality: 1,
                base64: true,
                exif: true
            });
        
            if (!result.canceled) {

                // image base64 only
                var base64 = result.assets[0].base64;

                // get image extension
                var extension = 'png'

                this.setState({image: base64, image_extension: extension})
            }else{
                console.log('***Product image selection cancelled***')
            }
        };

        this.setOpen1 = (open) => {
            this.setState({
                dropdown1_open: open
            });
        }

        this.setValue1 = (callback) => {
            this.setState(state => ({
                target_skin_types: callback(state.target_skin_types)
            }));
        }

        this.setItems1 = (callback) => {
            this.setState(state => ({
                skin_types: callback(state.skin_types)
            }));
        }

        this.setOpen2 = (open) => {
            this.setState({
                dropdown2_open: open
            });
        }

        this.setValue2 = (callback) => {
            this.setState(state => ({
                target_skin_disorders: callback(state.target_skin_disorders)
            }));
        }

        this.setItems2 = (callback) => {
            this.setState(state => ({
                skin_disorders: callback(state.skin_disorders)
            }));
        }

        this.GetBackendURL = async () => {
            var should_reload = false

            // get backend url
            let backend_url = await SecureStore.getItemAsync('backend_url');
            if (backend_url){
              this.setState({backend_url: backend_url})
            }else{ should_reload = true }

            if (should_reload == true){  const timeoutId = setTimeout(() => {this.GetBackendURL()}, 1000) }
        }

        this.GetUserDetails = async () => {
            var should_reload = false

            // get access token
            let token = await SecureStore.getItemAsync('token');
            if (token){
              this.setState({user_access_token: token})
            }else{ should_reload = true }

            // get username
            let user_name = await SecureStore.getItemAsync('user_name');
            if (user_name){
              this.setState({user_name: user_name})
            }else{ should_reload = true }

            if (should_reload == true){  const timeoutId = setTimeout(() => {this.GetUserDetails()}, 1000) }
        }
    };

    componentDidMount() {
        // get backend url
        this.GetBackendURL()
        // get user details
        this.GetUserDetails()
    }


    render() {

        return<View style={styles.container}>
            <ScrollView style={styles.scroll_view}>
                {
                    this.state.image == null
                    ? <View>
                        <View 
                            onStartShouldSetResponder={() => this.PickImage()}
                            style={{height: 170, backgroundColor: '#f2f2f2'}}
                        >
                            <View style={{marginLeft: 'auto', marginRight: 'auto', marginTop: 'auto', marginBottom: 'auto', borderRadius: 50, height: 90, width: 90, backgroundColor: '#eaeaea'}}>
                                <View style={{margin: 'auto'}}>
                                    <Feather name="image" size={28} color="grey" 
                                        style={{marginTop: 25, marginLeft: 'auto', marginRight: 'auto'}}
                                    />
                                </View>
                            </View>
                        </View>
                        <View style={{backgroundColor: '#f2f2f2', height: 40}}>
                            <Text style={{textAlign: 'center', marginTop: 'auto', marginBottom: 'auto'}}>
                                Select Image
                            </Text>
                        </View>
                    </View>
                    : <View
                        onStartShouldSetResponder={() => this.PickImage()}
                    >
                        <Text style={{color: '#00539C', fontWeight: 'bold', marginTop: 15, textAlign: 'center'}}>
                            Your selected image:
                        </Text>
                        <Image source={{uri: `data:image/png;base64,${this.state.image}`,}} style={{backgroundColor: '#f2f2f2', width: 300, height: 400, marginLeft: 'auto', marginRight: 'auto', marginTop: 20, marginBottom: 10, resizeMode: 'cover'}}/>
                    </View>
                }
                <View style={{marginLeft: 20, marginRight: 20}}>
                    <TextInput
                        // autoFocus={true}
                        onChangeText={(text) => this.HandleChange(text, "name")}
                        placeholder="Name"
                        placeholderTextColor='grey'
                        value={this.state.name}
                        style={{alignSelf: 'center', width: '100%', borderBottomColor: 'silver', borderBottomWidth: 1, marginTop: 50}}
                    />
                    <TextInput
                        // autoFocus={true}
                        onChangeText={(text) => this.HandleChange(text, "description")}
                        placeholder="Description"
                        placeholderTextColor='grey'
                        value={this.state.description}
                        style={{alignSelf: 'center', width: '100%', borderBottomColor: 'silver', borderBottomWidth: 1, marginTop: 40}}
                    />
                    <TextInput
                        // autoFocus={true}
                        keyboardType = 'number-pad'
                        onChangeText={(text) => this.HandleChange(text, "price")}
                        placeholder="Price"
                        placeholderTextColor='grey'
                        value={this.state.price}
                        style={{alignSelf: 'center', width: '100%', borderBottomColor: 'silver', borderBottomWidth: 1, marginTop: 40}}
                    />
                    <TextInput
                        // autoFocus={true}
                        onChangeText={(text) => this.HandleChange(text, "allergens")}
                        placeholder="Allergens (Comma separated eg Pollen, Eggs)"
                        placeholderTextColor='grey'
                        value={this.state.allergens}
                        style={{alignSelf: 'center', width: '100%', borderBottomColor: 'silver', borderBottomWidth: 1, marginTop: 40}}
                    />
                </View>
            </ScrollView>
            <View style={{marginLeft: 20, marginRight: 20}}>
                <Text style={{textAlign: 'left', color: '#3f4144', marginTop: 5}}>
                    Target skin types
                </Text>
                <DropDownPicker
                    // multiple={true}
                    // min={1}
                    // max={5}
                    open={this.state.dropdown1_open}
                    value={this.state.target_skin_types}
                    items={this.state.skin_types}
                    setOpen={this.setOpen1}
                    setValue={this.setValue1}
                    // setItems={this.setItems1}
                    style={{marginTop: 15}}
                />
                <Text style={{textAlign: 'left', color: '#3f4144', marginTop: 5}}>
                    Target skin disorders
                </Text>
                <DropDownPicker
                    // multiple={true}
                    // min={1}
                    // max={5}
                    open={this.state.dropdown2_open}
                    value={this.state.target_skin_disorders}
                    items={this.state.skin_disorders}
                    setOpen={this.setOpen2}
                    setValue={this.setValue2}
                    // setItems={this.setItems2}
                    style={{marginTop: 15, marginBottom: 70}}
                />
            </View>
            <View style={styles.bottom_bar}>
                <TouchableOpacity
                    key='add_product'
                    onPress={() => this.AddProduct()}
                    style={{backgroundColor: '#ffffff', marginLeft: 'auto', marginRight: 'auto', marginTop: 0, marginBottom: 0, width: '90%', height: 50, borderRadius: 5, borderWidth: 1, borderColor: '#00539C'}}
                >
                    <Text style={{textAlign: 'center', marginTop: 'auto', marginBottom: 'auto', fontWeight: 'bold', color: '#00539C', fontSize: 17}}>
                        Add Product
                    </Text>
                </TouchableOpacity>
            </View>
        </View>
    }

}

export default AddProducts;

const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: '#ffffff',
    //   textAlign: 'left',
    //   justifyContent: 'center'
    },
    header: {
        height: 40
    },
    search_bar: {
        // borderBottomWidth: 0.1,
        // borderBottomColor: '#eaeaea',
        flexDirection: 'row',
        backgroundColor: '#f2f2f2',
        // alignItems: 'center',
        // justifyContent: 'center',
        height: 50,
        marginTop: 50,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        top: 0
    },
    top_bar: {
        borderBottomWidth: 0.1,
        borderBottomColor: '#eaeaea',
        // flexDirection: 'row',
        backgroundColor: '#ffffff',
        // alignItems: 'center',
        // justifyContent: 'center',
        height: 100,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        top: 0
    },
    scroll_view: {
        height: 2
    },
    bottom_bar: {
        // flexDirection: 'row',
        backgroundColor: '#ffffff',
        alignItems: 'center',
        justifyContent: 'center',
        height: 70,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        bottom: 0
    }
});